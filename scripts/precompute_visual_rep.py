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
    chunk_size: int = 10000

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # fmt: on


@draccus.wrap()
def write_metadata(cfg: PrecomputeConfig) -> None:
    overwatch.info("Precomputing Visual Representations :: Starting")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load dataset annotations
    align_annotation_json, align_image_dir = cfg.dataset.align_stage_components
    ft_annotation_json, ft_image_dir = cfg.dataset.finetune_stage_components

    if align_annotation_json == ft_annotation_json:
        overwatch.info("Using same dataset for both align and finetune stages")
        annotation_path = cfg.dataset.dataset_root_dir / align_annotation_json
        image_base_dir = cfg.dataset.dataset_root_dir / align_image_dir
    else:
        overwatch.info("Using different datasets for align and finetune stages")
        annotation_path = cfg.dataset.dataset_root_dir / ft_annotation_json
        image_base_dir = cfg.dataset.dataset_root_dir / ft_image_dir

    overwatch.info(f"Loading dataset from {annotation_path}")
    with open(annotation_path, "r") as f:
        examples = json.load(f)

    total_examples = len(examples)
    num_chunks = (total_examples + cfg.chunk_size - 1) // cfg.chunk_size
    
    overwatch.info(f"Found {total_examples} examples")
    overwatch.info(f"Will process in {num_chunks} chunks of size {cfg.chunk_size}")

    # Save metadata for array jobs
    metadata = {
        "total_examples": total_examples,
        "num_chunks": num_chunks,
        "chunk_size": cfg.chunk_size,
        "annotation_path": str(annotation_path),
        "image_base_dir": str(image_base_dir),
        "model": {
            "vision_backbone_id": cfg.model.vision_backbone_id,
            "image_resize_strategy": cfg.model.image_resize_strategy,
        },
        "batch_size": cfg.batch_size,
        "device": cfg.device,
        "output_dir": str(cfg.output_dir),
    }
    
    metadata_file = cfg.output_dir / "array_job_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    overwatch.info(f"Saved metadata to {metadata_file}")


def process_chunk(chunk_id: int) -> None:
    """Process a single chunk of data."""
    
    metadata_file = Path("/share/data/speech/vlm_semantics/data/vision_features/array_job_metadata.json")
    if not metadata_file.exists():
        write_metadata()
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    chunk_size = metadata["chunk_size"]
    total_examples = metadata["total_examples"]
    start_idx = chunk_id * chunk_size # Calculate chunk boundaries
    end_idx = min(start_idx + chunk_size, total_examples)
    
    overwatch.info(f"Processing chunk {chunk_id}: examples {start_idx}-{end_idx-1}")
    
    with open(metadata["annotation_path"], "r") as f:
        examples = json.load(f)
    
    chunk_examples = examples[start_idx:end_idx]
    overwatch.info(f"Loaded {len(chunk_examples)} examples for chunk {chunk_id}")
    overwatch.info(f"Loading Vision Backbone: {metadata['model']['vision_backbone_id']}")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        metadata["model"]["vision_backbone_id"], 
        image_resize_strategy=metadata["model"]["image_resize_strategy"]
    )

    device = metadata["device"]
    vision_backbone = vision_backbone.to(device)
    vision_backbone.eval()
    
    chunk_features, chunk_ids, chunk_paths, chunk_conversations = [], [], [], []
    batch_images, batch_ids, batch_paths, batch_conversations = [], [], [], []
    
    image_base_dir = Path(metadata["image_base_dir"])
    batch_size = metadata["batch_size"]
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(chunk_examples, desc=f"Chunk {chunk_id}")):
            image_path = Path(example["image"])
            full_image_path = image_base_dir / image_path
            
            try:
                image = Image.open(full_image_path).convert("RGB")
                pixel_values = image_transform(image)
                
                batch_images.append(pixel_values)
                batch_ids.append(example.get("id", f"example_{start_idx + idx}"))
                batch_paths.append(str(image_path))
                batch_conversations.append(example["conversations"])
                
            except Exception as e:
                overwatch.warning(f"Failed to process image {full_image_path}: {e}")
                continue
            
            if len(batch_images) >= batch_size or idx == len(chunk_examples) - 1:
                if len(batch_images) > 0:
                    if isinstance(batch_images[0], dict):
                        batch_pixel_values = {
                            k: torch.stack([img[k] for img in batch_images]).to(device)
                            for k in batch_images[0].keys()
                        }
                    else:
                        batch_pixel_values = torch.stack(batch_images).to(device)
                    
                    patch_features = vision_backbone(batch_pixel_values)
                    
                    for i, (example_id, img_path, conversation) in enumerate(zip(batch_ids, batch_paths, batch_conversations)):
                        chunk_features.append(patch_features[i].cpu())
                        chunk_ids.append(example_id)
                        chunk_paths.append(img_path)
                        chunk_conversations.append(conversation)
                    
                    batch_images = []
                    batch_ids = []
                    batch_paths = []
                    batch_conversations = []
    
    output_dir = Path(metadata["output_dir"])
    chunk_file = output_dir / f"chunk_{chunk_id:04d}.pt"
    
    chunk_data = {
        "patch_features": torch.stack(chunk_features),
        "ids": chunk_ids,
        "image_paths": chunk_paths,
        "conversations": chunk_conversations,
        "chunk_info": {
            "chunk_id": chunk_id,
            "start_idx": start_idx,
            "end_idx": end_idx - 1,
            "num_examples": len(chunk_features),
        }
    }
    
    torch.save(chunk_data, chunk_file)
    overwatch.info(f"Saved chunk {chunk_id} to {chunk_file}")
    overwatch.info(f"Processed {len(chunk_features)} examples in chunk {chunk_id}")


if __name__ == "__main__":
    chunk_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    process_chunk(chunk_id)