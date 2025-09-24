"""
resumable_pretrain.py

Modified version of pretrain.py with resumption capability.
Saves and loads training state including epoch, step, and dataset position.
The 'save_checkpoint' and 'load_checkpoint' methods are in the specific training strategy.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing import get_dataset_and_collator
from prismatic.training import Metrics, get_train_strategy
from prismatic.util import set_global_seed

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class ResumablePretrainConfig:
    # fmt: off

    # ModelConfig (`prismatic/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.PRISM_DINOSIGLIP_7B.model_id)
    )

    # DatasetConfig (`prismatic/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id)
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    stage: str = "align"                                            # Pretraining Stage
    
    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("wandb",)                          # Trackers to initialize
    wandb_project: str = "prismatic-vlms"
    wandb_entity: str = "sallyxu"

    resume_from_checkpoint: Optional[Path] = None                   # Checkpoint to resume from (full state)
    reset_for_new_stage: Optional[bool] = field(default=False)      # Whether to reset epoch/step counters when starting a new stage

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage`."""
        if self.stage == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

        elif self.stage.endswith("finetune"):
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")


class ResumableDataset:
    """Wrapper around dataset that can skip to a specific sample index."""
    
    def __init__(self, dataset, start_index=0):
        self.dataset = dataset
        self.start_index = start_index
        self.current_index = start_index
    
    def __len__(self):
        return len(self.dataset) - self.start_index
    
    def __getitem__(self, idx):
        actual_idx = (self.start_index + idx) % len(self.dataset)
        return self.dataset[actual_idx]
    
    def get_state(self):
        return {"current_index": self.current_index, "start_index": self.start_index}
    
    def load_state(self, state):
        self.current_index = state["current_index"]
        self.start_index = state["start_index"]
        

@draccus.wrap()
def resumable_pretrain(cfg: ResumablePretrainConfig) -> None:
    # print(f"DEBUG: cfg.resume_from_checkpoint = {cfg.resume_from_checkpoint}")
    overwatch.info("Resumable Prismatic VLM Training :: Gathering Light")

    # Setup distributed training
    torch.cuda.set_device(device_id := (overwatch.local_rank()))
    torch.cuda.empty_cache()

    # Create run directory
    model_id = cfg.model.model_id
    dataset_id = cfg.dataset.dataset_id
    if dataset_id == "llava-v15":
        cfg.run_id = f"{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    else:
        cfg.run_id = f"{dataset_id}+{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    # Setup
    overwatch.info('"Life is like a prism; what you see depends on how you turn the glass."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    run_dir = cfg.run_root_dir / cfg.run_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_dir / "checkpoints", exist_ok=True)

    # Save config
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load model components
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )

    overwatch.info(f"Loading LLM Backbone [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token
    )

    # Create VLM
    overwatch.info(f"Instantiating PrismaticVLM `{model_id}` for Training Stage = `{cfg.stage}`")
    vlm = get_vlm(
        model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    )

    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.freeze_backbones(cfg.stage)

    # Get dataset
    overwatch.info(f"Loading training dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")
    train_dataset, collator = get_dataset_and_collator(
        cfg.stage,
        cfg.dataset,
        image_transform,
        tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        default_image_resolution=vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side,
    )
    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy: {cfg.train_strategy}")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))

    # Wrap dataset for resumption (simplified - strategy handles the details)
    resumable_dataset = train_dataset

    # Create metrics
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        cfg.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Run training with resumption support
    overwatch.info("Starting Training Loop")
    resume_checkpoint = cfg.resume_from_checkpoint if cfg.resume_from_checkpoint else None
    reset_for_new_stage = cfg.reset_for_new_stage
    is_resumable_strategy = hasattr(train_strategy, 'load_checkpoint') and hasattr(train_strategy, 'resume_epoch')
    
    if is_resumable_strategy:
        overwatch.info("Using resumable training strategy")
        train_strategy.run_training(
            resumable_dataset, collator, metrics, 
            stage=cfg.stage, seed=cfg.seed, resume_checkpoint=resume_checkpoint, reset_for_new_stage=reset_for_new_stage
        )
    else:
        overwatch.info("[WARNING] Training Strategy is Not Resumable. Stopping...")

    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    resumable_pretrain()
