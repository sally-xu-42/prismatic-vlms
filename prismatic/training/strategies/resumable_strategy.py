"""
resumable_strategy.py

Resumable training strategy that inherits from the base TrainingStrategy and adds
full resumption capabilities including epoch, step, optimizer state, and dataset position.

Resumable Training Strategies (ResDDP, ResFSDP-Grad, ResFSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.overwatch import initialize_overwatch
from prismatic.conf.datasets import DatasetRegistry
from prismatic.training.strategies.base_strategy import TrainingStrategy
from prismatic.training.metrics import Metrics
from prismatic.models import get_vlm
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

sys.path.insert(0, str("/share/data/speech/txu/vlm_semantics"))

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
        # Track resumption state in addition to TrainingStrategy attributes
        self.resume_epoch = 0
        self.resume_step = 0
        self.resume_samples_seen = 0
        self.reset_for_new_stage = False  # Default to False: Whether to reset counters for a new training stage

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
        resume_checkpoint: Optional[Path] = None,
        reset_for_new_stage: Optional[bool] = False,
    ) -> None:
        """Enhanced training loop with resumption support."""
        
        # Load checkpoint if resuming
        if resume_checkpoint:
            # Reset counters if starting a new stage
            self.reset_for_new_stage = reset_for_new_stage
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
                overwatch.info(f"Starting epoch {epoch}...")

                # Calculate batches to skip if resuming mid-epoch
                batch_skip_count = 0
                if epoch == self.resume_epoch and resume_dataset_offset > 0:
                    effective_batch_size = self.per_device_batch_size * overwatch.world_size()
                    batch_skip_count = resume_dataset_offset // effective_batch_size
                    overwatch.info(f"Skipping {batch_skip_count} batches in epoch {epoch}")

                for train_idx, batch in enumerate(dataloader):
                    # Skip batches if resuming mid-epoch
                    # TODO: This is very inefficient if resumed in mid-epoch; temporary workaround
                    if epoch == self.resume_epoch and train_idx < batch_skip_count:
                        continue
                    # print(f"[Debug] patch_features before training loop: {batch['patch_features']}") # ===================>> support for precomputed patch features
                    # Forward pass (same as base class)
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        # print(f"[Debug] patch_features in training loop: {batch['patch_features']}") # ===================>> support for precomputed patch features
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            # image_file_names=batch["image_file_names"], # ===================>> support for precomputed patch features
                            patch_features=batch["patch_features"], # ===================>> support for precomputed patch features
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss
                        print(f"Input ids shape: {batch['input_ids'][0].shape}")
                        print(f"logits shape: {output.logits.shape}")
                        # Print logits for the token where the model is predicting
                        last_token_logits = output.logits[0, -3, :]  # [vocab_size]
                        top_k_logits, top_k_indices = torch.topk(last_token_logits, k=5)
                        print(f"Top 5 logits for last token: {top_k_logits}")
                        print(f"Top 5 token IDs: {top_k_indices}")

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
                        torch.cuda.empty_cache() # ====================>> Added to reduce memory fragmentation

                        # Add checkpoint saving and logging every 500 steps
                        if metrics.global_step % 500 == 0:
                            self.save_checkpoint(
                                run_dir=metrics.run_dir, global_step=metrics.global_step, epoch=epoch, stage=stage,
                                train_loss=loss.item(), samples_seen=samples_seen
                            )
                            status = metrics.push()
                            overwatch.info(
                                f"Step {metrics.global_step}, Loss: {loss.item():.4f}, \
                                LR: {self.lr_scheduler.get_last_lr()[0]:.4f}"
                            )

                        # Check for Termination
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(
                                run_dir=metrics.run_dir, global_step=metrics.global_step, epoch=epoch, stage=stage,
                                train_loss=loss.item(), samples_seen=samples_seen
                            )
                            dist.barrier()
                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end of training
            if self.max_steps is None:
                self.save_checkpoint(
                    run_dir=metrics.run_dir, global_step=metrics.global_step, epoch=epoch, stage=stage,
                    train_loss=loss.item(), samples_seen=samples_seen
                )
                dist.barrier()
