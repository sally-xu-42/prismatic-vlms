"""
precompute_visual_rep.py

Precomputes and saves visual representations from the vision encoder.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import draccus
import torch
from PIL import Image
from tqdm import tqdm

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from prismatic.models import get_vision_backbone_and_transform
from prismatic.overwatch import initialize_overwatch

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PrecomputeConfig:
    # fmt: off

    # ModelConfig (`prismatic/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.PRISM_DINOSIGLIP_7B.model_id)
    )

    # DatasetConfig (`prismatic/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.CLEVR.dataset_id)
    )
    
    # Output directory for saved representations
    output_dir: Path = Path("/share/data/speech/vlm_semantics/data/vision_features")
    batch_size: int = 24

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # fmt: on


@draccus.wrap()
def precompute_visual_representations(cfg: PrecomputeConfig) -> None:
    # Setup
    overwatch.info("Precomputing Visual Representations :: Starting")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load vision backbone and transform
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )
    
    # Move vision backbone to device and set to eval mode
    vision_backbone = vision_backbone.to(cfg.device)
    vision_backbone.eval()

    # Load dataset annotations
    align_annotation_json, align_image_dir = cfg.dataset.align_stage_components
    ft_annotation_json, ft_image_dir = cfg.dataset.finetune_stage_components

    if align_annotation_json == ft_annotation_json:
        overwatch.info("Using same dataset for both align and finetune stages")
        overwatch.info(f"Using annotation file: {align_annotation_json}, only compute once")
        annotation_path = cfg.dataset.dataset_root_dir / align_annotation_json
        image_base_dir = cfg.dataset.dataset_root_dir / align_image_dir
    else:
        overwatch.info("Using different datasets for align and finetune stages")
        overwatch.info(f"Using annotation files: {align_annotation_json} and {ft_annotation_json}")
        annotation_path = cfg.dataset.dataset_root_dir / ft_annotation_json
        image_base_dir = cfg.dataset.dataset_root_dir / ft_image_dir

    overwatch.info(f"Loading dataset from {annotation_path}")
    with open(annotation_path, "r") as f:
        examples = json.load(f)

    overwatch.info(f"Found {len(examples)} examples")

    # Collect all features
    all_features = []
    all_ids = []
    all_image_paths = []
    all_conversations = []
    
    batch_images = []
    batch_ids = []
    batch_paths = []
    batch_conversations = []
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(examples, desc="Extracting features")):
            image_path = Path(example["image"])
            full_image_path = image_base_dir / image_path
            
            try:
                image = Image.open(full_image_path).convert("RGB")
                pixel_values = image_transform(image)
                
                batch_images.append(pixel_values)
                batch_ids.append(example.get("id", f"example_{idx}"))
                batch_paths.append(str(image_path))
                batch_conversations.append(example["conversations"])
                
            except Exception as e:
                overwatch.warning(f"Failed to process image {full_image_path}: {e}")
                continue
            
            # Process batch when full or at end
            if len(batch_images) >= cfg.batch_size or idx == len(examples) - 1:
                if len(batch_images) > 0:
                    # Stack batch
                    if isinstance(batch_images[0], dict):
                        # Fused backbone case
                        batch_pixel_values = {
                            k: torch.stack([img[k] for img in batch_images]).to(cfg.device)
                            for k in batch_images[0].keys()
                        }
                    else:
                        # Single backbone case
                        batch_pixel_values = torch.stack(batch_images).to(cfg.device)
                    
                    # Extract features
                    patch_features = vision_backbone(batch_pixel_values)
                    
                    # Store features and metadata
                    for i, (example_id, img_path, conversation) in enumerate(zip(batch_ids, batch_paths, batch_conversations)):
                        all_features.append(patch_features[i].cpu())
                        all_ids.append(example_id)
                        all_image_paths.append(img_path)
                        all_conversations.append(conversation)
                    
                    # Clear batch
                    batch_images = []
                    batch_ids = []
                    batch_paths = []
                    batch_conversations = []

    # Stack all features into a single tensor
    overwatch.info("Stacking all features...")
    stacked_features = torch.stack(all_features)  # Shape: [num_examples, num_patches, embed_dim]
    
    precomputed_data = {
        "patch_features": stacked_features,  # [num_examples, num_patches, embed_dim]
        "ids": all_ids,
        "image_paths": all_image_paths,
        "conversations": all_conversations,
        "metadata": {
            "num_examples": len(all_features),
            "vision_backbone_id": cfg.model.vision_backbone_id,
            "image_resize_strategy": cfg.model.image_resize_strategy,
            "embed_dim": vision_backbone.embed_dim,
            "num_patches": vision_backbone.num_patches,
            "default_image_resolution": vision_backbone.default_image_resolution,
            "feature_shape": list(stacked_features.shape),
        }
    }
    
    # Save all features in a single file
    output_file = cfg.output_dir / "precomputed_features.pt"
    overwatch.info(f"Saving all features to {output_file}")
    torch.save(precomputed_data, output_file)
    
    overwatch.info(f"Saved {len(all_features)} examples with features shape {stacked_features.shape}")
    overwatch.info("... and that's all, folks!")


if __name__ == "__main__":
    precompute_visual_representations()