"""
resumable_strategy.py

Resumable training strategy that inherits from the base TrainingStrategy and adds
full resumption capabilities including epoch, step, optimizer state, and dataset position.

Resumable Training Strategies (ResDDP, ResFSDP-Grad, ResFSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import TrainingStrategy
from prismatic.training.metrics import Metrics
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# === Abstract Base Class for an arbitrary Resumable Training Strategy ===
class ResumableTrainingStrategy(TrainingStrategy):
    """
    Enhanced training strategy with resumption capabilities.
    Inherits from base TrainingStrategy and overrides save_checkpoint and run_training.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track resumption state
        self.resume_epoch = 0
        self.resume_step = 0
        self.resume_samples_seen = 0

    def save_full_checkpoint(
        self,
        checkpoint_path: Path,
        model_state_dicts: dict,
        global_step: int,
        epoch: int,
        samples_seen: int,
        train_loss: Optional[float] = None,
    ) -> None:
        """Save a complete checkpoint with all training state. Overrided for FSDP due to state mismatch."""
        checkpoint = {
            "model": model_state_dicts,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "epoch": epoch,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "rng_state": torch.get_rng_state(),
            "train_loss": train_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        overwatch.info(f"Saved resumable checkpoint: step={global_step}, epoch={epoch}, samples={samples_seen}")

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load full training state from checkpoint."""
        overwatch.info(f"Loading resumable checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model weights - handle both old and new checkpoint formats
        if "model" in checkpoint:
            model_dict = checkpoint["model"]
            for key in self.trainable_module_keys:
                if key in model_dict:
                    # Handle wrapped models (DDP/FSDP)
                    if hasattr(self.vlm, 'module'):
                        getattr(self.vlm.module, key).load_state_dict(model_dict[key])
                    else:
                        getattr(self.vlm, key).load_state_dict(model_dict[key])
        
        # Load optimizer state
        # if "optimizer" in checkpoint and self.optimizer:
        #     self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Load scheduler state
        if "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
        # Restore RNG state
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        
        # Set resumption state
        self.resume_epoch = checkpoint.get("epoch", 0)
        self.resume_step = checkpoint.get("global_step", 0)
        self.resume_samples_seen = checkpoint.get("samples_seen", 0)
        
        overwatch.info(f"Resumed from: epoch={self.resume_epoch}, step={self.resume_step}, samples={self.resume_samples_seen}")
        
        return {
            "epoch": self.resume_epoch,
            "global_step": self.resume_step,
            "samples_seen": self.resume_samples_seen,
        }

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
        resume_checkpoint: Optional[Path] = None,
    ) -> None:
        """Enhanced training loop with resumption support."""
        
        # Load checkpoint if resuming
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
            # Update metrics to start from correct step
            metrics.global_step = self.resume_step
            overwatch.info(f"DEBUG: Final check - metrics.global_step = {metrics.global_step}")

        # Create sampler (same as base class)
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            self.epochs = 100

        # Calculate resumption parameters
        samples_per_epoch = len(dataset)
        resume_dataset_offset = self.resume_samples_seen % samples_per_epoch if self.resume_samples_seen > 0 else 0
        
        # Log resumption info
        if resume_dataset_offset > 0:
            overwatch.info(f"Resuming: skipping {resume_dataset_offset} samples in dataset")

        # === Enhanced Training Loop ===
        status = metrics.get_status()
        total_steps = (self.epochs * steps_per_epoch) if self.max_steps is None else self.max_steps
        
        with tqdm(
            total=total_steps,
            initial=self.resume_step,  # Start progress bar from resume point
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            
            samples_seen = self.resume_samples_seen
            
            for epoch in range(self.resume_epoch, self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)
                self.optimizer.zero_grad()

                # Calculate batches to skip if resuming mid-epoch
                batch_skip_count = 0
                if epoch == self.resume_epoch and resume_dataset_offset > 0:
                    effective_batch_size = self.per_device_batch_size * overwatch.world_size()
                    batch_skip_count = resume_dataset_offset // effective_batch_size
                    overwatch.info(f"Skipping {batch_skip_count} batches in epoch {epoch}")

                for train_idx, batch in enumerate(dataloader):
                    # Skip batches if resuming mid-epoch
                    if epoch == self.resume_epoch and train_idx < batch_skip_count:
                        continue

                    # Forward pass (same as base class)
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Track samples processed
                    batch_size = batch["input_ids"].size(0)
                    samples_seen += batch_size * overwatch.world_size()

                    # Commit Loss (same as base class)
                    metrics.commit(loss=loss)

                    # Backward pass (same as base class)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Enhanced checkpoint saving with samples_seen
                        if metrics.global_step % 500 == 0:
                            self.save_checkpoint(
                                metrics.run_dir, metrics.global_step, epoch, 
                                loss.item(), samples_seen=samples_seen
                            )

                        # Check for Termination
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(
                                metrics.run_dir, metrics.global_step, epoch, 
                                loss.item(), samples_seen=samples_seen
                            )
                            dist.barrier()
                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end of training
            if self.max_steps is None:
                self.save_checkpoint(
                    metrics.run_dir, metrics.global_step, epoch, 
                    loss.item(), samples_seen=samples_seen
                )
                dist.barrier()