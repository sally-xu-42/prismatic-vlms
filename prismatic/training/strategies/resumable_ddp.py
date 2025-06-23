"""
resumable_ddp.py

Resumable DDP strategy that combines DDPStrategy with ResumableTrainingStrategy.
"""

import shutil
from pathlib import Path
from typing import Optional

import torch

from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.ddp import DDPStrategy
from prismatic.training.strategies.resumable_strategy import ResumableTrainingStrategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class ResumableDDPStrategy(ResumableTrainingStrategy, DDPStrategy):
    """
    Resumable DDP Strategy that combines:
    - DDPStrategy: for DDP-specific setup, gradient clipping, etc.
    - ResumableTrainingStrategy: for enhanced training loop and checkpoint management
    """
    
    def __init__(self, *args, **kwargs):
        # Initialize both parent classes
        super().__init__(*args, **kwargs)

    @overwatch.rank_zero_only
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
        samples_seen: int = 0,
    ) -> None:
        """Enhanced checkpoint saving with full training state."""
        assert hasattr(self.vlm, 'module'), "save_checkpoint assumes VLM is wrapped (DDP/FSDP)!"

        # Get model state dictionaries
        model_state_dicts = {
            mkey: getattr(self.vlm.module, mkey).state_dict()
            for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
        }

        # Set checkpoint path (same naming as original)
        checkpoint_dir = run_dir / "checkpoints"
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        # Save full checkpoint with training state
        self.save_full_checkpoint(
            checkpoint_path, model_state_dicts, global_step, epoch, samples_seen, train_loss
        )
        
        # Copy to latest-checkpoint.pt (for easy resumption)
        shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

    # Inherit run_setup and clip_grad_norm from DDPStrategy
    # Inherit run_training and load_checkpoint from ResumableTrainingStrategy