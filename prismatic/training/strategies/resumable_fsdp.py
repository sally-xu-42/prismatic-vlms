"""
resumable_fsdp.py

Resumable FSDP strategy that combines FSDPStrategy with ResumableTrainingStrategy.
Basically we have to rewrite the `save_checkpoint` and add `load_checkpoint` methods.
This strategy allows for saving and loading checkpoints with FSDP-specific handling,
"""

import os
import re
import json
from tqdm import tqdm
from draccus import decode
from src import EvalDataset, PaddedCollatorForEval, get_dataset_and_collator

import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.fsdp import FSDPStrategy
from prismatic.training.strategies.resumable_strategy import ResumableTrainingStrategy
from prismatic.conf import ModelConfig, DatasetConfig, DatasetRegistry
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from prismatic.models.backbones.vision import ImageTransform

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class ResumableFSDPStrategy(ResumableTrainingStrategy, FSDPStrategy):
    """
    Resumable FSDP Strategy that inhibits from both `FSDPStrategy` and `ResumableTrainingStrategy`:
    - FSDPStrategy: for FSDP-specific setup, gradient clipping, etc.
    - ResumableTrainingStrategy: for enhanced training loop and checkpoint management
    Main methods are `save_checkpoint`, `load_checkpoint` and `eval_latest`, which handle FSDP-specific state dicts.
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        super().__init__(*args, **kwargs)

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        stage: str,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
        samples_seen: int = 0,
    ) -> None:
        """Overrides FSDPstrategy. Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, FSDP), "FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"

        # Summon Full State Dictionary =>> Reconstitute from Shards
        with FSDP.state_dict_type(self.vlm, self.fsdp_state_dict_type, self.fsdp_save_policy):
            full_vlm_state_dict = self.vlm.state_dict()
            # By default, we only save trainable parameters
            model_state_dicts = {
                mkey: OrderedDict() for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
            }

            # Iterate through `full_vlm_state_dict` and split `mkey.{full_dotted_path}` -> `mkey: {full_dotted_path}`
            for key, param in full_vlm_state_dict.items():
                for mkey in model_state_dicts:
                    if key.startswith(mprefix := f"{mkey}."):
                        model_state_dicts[mkey][key.removeprefix(mprefix)] = param

            # Save on rank zero *only*
            if overwatch.is_rank_zero():
                checkpoint_dir = run_dir / "checkpoints"
                if train_loss is None:
                    checkpoint_path = checkpoint_dir / f"{stage}-step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
                else:
                    checkpoint_path = (
                        checkpoint_dir / f"{stage}-step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
                    )
                # optimizer_state = self.optimizer.state_dict()
                # optimizer_state = FSDP.full_optim_state_dict(self.vlm, self.optimizer, 
                #                                             rank0_only=True)
                checkpoint = {
                    "model": model_state_dicts,
                    # "optimizer": optimizer_state,  # Saved with FSDP context
                    "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    "epoch": epoch,
                    "global_step": global_step,
                    "samples_seen": samples_seen,
                    "rng_state": torch.get_rng_state(),
                    "train_loss": train_loss,
                }
                # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
                torch.save(checkpoint, checkpoint_path)
                shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")
                overwatch.info(f"Saved FSDP resumable checkpoint: step={global_step}, epoch={epoch}, samples={samples_seen}")

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        """Enhanced checkpoint loading for FSDP with proper state dict handling."""
        overwatch.info(f"Loading resumable FSDP checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model weights with FSDP state dict context
        if "model" in checkpoint:
            model_dict = checkpoint["model"]
            # For FSDP, we need to reconstruct the full state dict and load it properly
            with FSDP.state_dict_type(self.vlm, self.fsdp_state_dict_type, self.fsdp_save_policy):
                # Reconstruct full state dict from module-based format
                full_state_dict = {}
                for mkey, module_state in model_dict.items():
                    for param_key, param_value in module_state.items():
                        full_key = f"{mkey}.{param_key}"
                        full_state_dict[full_key] = param_value
                
                # Load the reconstructed state dict
                self.vlm.load_state_dict(full_state_dict, strict=False)
        
        # # Load optimizer state (IMPORTANT: mismatch)
        # if "optimizer" in checkpoint and self.optimizer:
        #     try:
        #         full_optim = checkpoint["optimizer"]
        #         local_optim = FSDP.optim_state_dict_to_load(
        #             optim_state_dict=full_optim,
        #             model=self.vlm,
        #             optim=self.optimizer
        #         )
        #         self.optimizer.load_state_dict(local_optim)
        #     except Exception as e:
        #         overwatch.warning(f"Failed to load optimizer state: {e}")
        
        # Load scheduler state
        if (not self.reset_for_new_stage) and "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
        # Restore RNG state
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        
        # Set resumption state
        if not self.reset_for_new_stage:
            self.resume_epoch = checkpoint.get("epoch", 0)
            self.resume_step = checkpoint.get("global_step", 0)
            self.resume_samples_seen = checkpoint.get("samples_seen", 0)
        else:
            self.resume_epoch, self.resume_step, self.resume_samples_seen = 0, 0, 0
        
        overwatch.info(f"Resumed FSDP from: epoch={self.resume_epoch}, step={self.resume_step}, samples={self.resume_samples_seen}")
        
        return {
            "epoch": self.resume_epoch,
            "global_step": self.resume_step,
            "samples_seen": self.resume_samples_seen,
        }

    # def eval_latest(self, run_dir: Path) -> dict:
    #     """Evaluate on the latest checkpoint from the run directory DURING TRAINING.
    #     Because of hf and FSDP incompatibility issues, we have to load the model from the config file and the latest checkpoint.
    #     This is a workaround to ensure we can evaluate the model without loading the entire FSDP state dict.
    #     """
    #     self.vlm.cpu() # shove shards to CPUs
    #     torch.cuda.empty_cache()
    #     config_path = os.path.join(os.getenv("RUN_DIR", run_dir), "config.json") if os.path.isdir(os.getenv("RUN_DIR", run_dir)) else None
    #     print(f"Loading config {config_path}...")
    #     with open(config_path, 'r') as f:
    #         config = json.load(f)
    #         cfg = decode(ModelConfig, config["model"])
    #         print(f"{cfg}\n")

    #     vision_backbone, image_transform = get_vision_backbone_and_transform(
    #         cfg.vision_backbone_id,
    #         image_resize_strategy=cfg.image_resize_strategy,
    #     )
    #     llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
    #         cfg.llm_backbone_id, 
    #         llm_max_length=cfg.llm_max_length,
    #         hf_token=config["hf_token"]
    #     )

    #     vlm = get_vlm(
    #         cfg.model_id,
    #         cfg.arch_specifier,
    #         vision_backbone,
    #         llm_backbone,
    #         enable_mixed_precision_training=cfg.enable_mixed_precision_training,
    #     )
    #     vlm.to(torch.cuda.current_device())
        
    #     checkpoint_path = os.path.join(os.getenv("RUN_DIR", run_dir), "checkpoints", "latest-checkpoint.pt") if os.path.isdir(os.getenv("RUN_DIR", run_dir)) else None
    #     overwatch.info(f"\nEvaluating latest checkpoint from {checkpoint_path}...")
    #     checkpoint = torch.load(checkpoint_path, map_location="cuda")
    #     vlm.projector.load_state_dict(checkpoint["model"]["projector"])
    #     vlm.projector.to(torch.cuda.current_device())
    #     vlm.eval()  # Set model to evaluation mode

    #     # We temporarily evaluate on the training split accuracy to ensure the models are learning
    #     dataset_cfg = decode(DatasetConfig, config["dataset"])
    #     # # The evaluation dataset configuration, pure auto-generated questions
    #     # dataset_cfg = DatasetRegistry.CLEVR.value

    #     val_dataset, eval_collator = get_dataset_and_collator(
    #         dataset_cfg=dataset_cfg,
    #         image_transform=image_transform,
    #         tokenizer=tokenizer,
    #         default_image_resolution=vision_backbone.default_image_resolution,
    #         padding_side=tokenizer.padding_side
    #     )
    #     dataloader = torch.utils.data.DataLoader(
    #         val_dataset,
    #         batch_size=100,
    #         collate_fn=eval_collator,
    #         shuffle=True,
    #         num_workers=1
    #     )
    #     with torch.no_grad():
    #         for batch in tqdm(dataloader, desc="Processing batches"):
    #             correct = 0
    #             images = batch["image"]
    #             prompts = batch['input_text']
    #             answers = batch['answer']
    #             batch_size = len(prompts)
    #             # EvalDataset returns: pixel_values, input_text, answer, input_ids, labels, image
    #             for i in range(batch_size):
    #                 # Using the new vlm
    #                 output = vlm.generate(
    #                     images[i],
    #                     prompts[i],
    #                     max_new_tokens=10,
    #                     temperature=None
    #                 )
    #                 predicted = output.strip().lower()
    #                 predicted = re.sub(r'[^\w\s]', '', predicted).strip()
    #                 ground_truth = answers[i].strip().lower()
    #                 if ground_truth in predicted:
    #                     correct += 1
    #                     # print(f"Correct: {predicted} == {ground_truth}")
    #             accuracy = correct / batch_size if batch_size > 0 else 0.0
    #             overwatch.info(f"{correct} correct questions in {batch_size} total questions\n")

    #             # cleanup memory
    #             del images, prompts, answers, output, predicted, ground_truth
    #             del dataloader, batch
    #             del vlm, llm_backbone, vision_backbone, tokenizer, image_transform
    #             torch.cuda.empty_cache()
                
    #             self.vlm.cuda()
    #             return accuracy

    # Inherit run_setup from FSDPStrategy
    # Inherit clip_grad_norm from FSDPStrategy  
    # Inherit run_training from ResumableTrainingStrategy