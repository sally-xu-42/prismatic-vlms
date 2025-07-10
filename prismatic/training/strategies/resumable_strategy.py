"""
resumable_strategy.py

Resumable training strategy that inherits from the base TrainingStrategy and adds
full resumption capabilities including epoch, step, optimizer state, and dataset position.

Resumable Training Strategies (ResDDP, ResFSDP-Grad, ResFSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""
import sys
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
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling

sys.path.insert(0, str("/share/data/speech/txu/vlm_semantics"))
from src import EvalDataset, PaddedCollatorForEval, get_dataset_and_collator

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
    
    def calculate_validation_loss(
            self,
            val_dataset: Dataset,
            collator: PaddedCollatorForLanguageModeling,
            seed: int = 7
    ) -> torch.Tensor:
        """Calculate validation loss during training."""
        sampler = DistributedSampler(
            val_dataset,
            num_replicas=overwatch.world_size(),
            rank=overwatch.rank(),
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        dataloader = DataLoader(
            val_dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )
        self.vlm.eval()
        # validtion loss over a random batch
        with torch.no_grad():
            for batch in dataloader:
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
                    return loss
    
    def calculate_accuracy(
            self,
            seed: int = 7,
    ) -> torch.Tensor:
        """Calculate validation accuracy on a single random batch."""
        # The evaluation dataset configuration, pure auto-generated questions
        dataset_cfg = DatasetRegistry.CLEVR.value
        val_dataset, eval_collator = get_dataset_and_collator(
            dataset_cfg=dataset_cfg,
            image_transform=self.vlm.vision_backbone.image_transform,
            tokenizer=self.vlm.llm_backbone.tokenizer,
            default_image_resolution=self.vlm.vision_backbone.default_image_resolution,
            padding_side=self.vlm.llm_backbone.tokenizer.padding_side
        )
        sampler = DistributedSampler(
            val_dataset,
            num_replicas=overwatch.world_size(),
            rank=overwatch.rank(),
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        dataloader = DataLoader(
            val_dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=eval_collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )
        
        self.vlm.eval()
        with torch.no_grad():
            for batch in dataloader:
                correct = 0
                total = len(batch['answer'])
                for i in range(total):
                    output = self.vlm.generate(
                        batch["image"][i],
                        batch['input_text'][i],
                        max_new_tokens=10,
                        temperature=None
                    )
                    predicted = output.strip().lower()
                    ground_truth = batch['answer'][i].strip().lower()
                    if ground_truth in predicted:
                        correct += 1
                        print(f"Correct: {predicted} == {ground_truth}")
                accuracy = correct / total if total > 0 else 0.0
                print(f"{correct} correct questions in {total} total questions")
                return accuracy

    def run_training(
        self,
        dataset: Dataset,
        val_dataset: Dataset,
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

                        # Add checkpoint saving and logging every 500 steps
                        if metrics.global_step % 50 == 0:
                            self.save_checkpoint(
                                metrics.run_dir, metrics.global_step, epoch, 
                                loss.item(), samples_seen=samples_seen
                            )
                            # val_loss = self.calculate_validation_loss(
                            #     val_dataset, collator, seed=seed
                            # )
                            val_accuracy = self.calculate_accuracy(seed=seed)
                            metrics.commit(validation_loss=val_accuracy)
                            status = metrics.push()
                            overwatch.info(
                                f"Step {metrics.global_step}, Loss: {loss.item():.4f}, \
                                LR: {self.lr_scheduler.get_last_lr()[0]:.4f}, \
                                Validation Accuracy: {val_accuracy:.4f}"
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