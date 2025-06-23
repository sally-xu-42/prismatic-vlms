"""
materialize.py

Factory class defining functions for instantiating various Training Strategies, supporting different VLMs, backbones,
and strategy configurations.
"""

from typing import Callable, Optional

import torch

from prismatic.models.vlms import PrismaticVLM
from prismatic.training.strategies import ResumableTrainingStrategy, ResumableFSDPStrategy

# Registry =>> Maps ID --> {cls(), kwargs} :: supports both resumable and original strategies
TRAIN_STRATEGIES = {
    "fsdp-shard-grad-op": {"cls": ResumableFSDPStrategy, "kwargs": {"sharding_strategy": "shard-grad-op"}},
    "fsdp-full-shard": {"cls": ResumableFSDPStrategy, "kwargs": {"sharding_strategy": "full-shard"}}
}


def get_train_strategy(
    train_strategy: str,
    vlm: PrismaticVLM,
    device_id: int,
    epochs: int,
    max_steps: Optional[int],
    global_batch_size: int,
    per_device_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    enable_gradient_checkpointing: bool = True,
    enable_mixed_precision_training: bool = True,
    reduce_in_full_precision: bool = False,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> ResumableTrainingStrategy:
    """
    Factory function to get training strategy.
    
    Args:
        train_strategy: Strategy identifier from TRAIN_STRATEGIES registry
        ... (other training parameters)
    
    Returns:
        TrainingStrategy: Configured training strategy instance
        
    Available strategies:
        - "ddp": Resumable DDP strategy (recommended)
        - "fsdp-shard-grad-op": Resumable FSDP with gradient/optimizer sharding
        - "fsdp-full-shard": Resumable FSDP with full parameter sharding  
        - "ddp-original": Original DDP strategy (no resumption)
        - "fsdp-*-original": Original FSDP strategies (no resumption)
    """
    if train_strategy in TRAIN_STRATEGIES:
        strategy_cfg = TRAIN_STRATEGIES[train_strategy]
        strategy = strategy_cfg["cls"](
            vlm=vlm,
            device_id=device_id,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
            **strategy_cfg["kwargs"],
        )
        return strategy
    else:
        available_strategies = list(TRAIN_STRATEGIES.keys())
        raise ValueError(
            f"Train Strategy `{train_strategy}` is not supported!\n"
            f"Available strategies: {available_strategies}"
        )