"""
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant

    # Dataset Components for each Stage in < align | finetune >
    align_stage_components: Tuple[Path, Path]       # Path to annotation file and images directory for `align` stage
    finetune_stage_components: Tuple[Path, Path]    # Path to annotation file and images directory for `finetune` stage

    dataset_root_dir: Path                          # Path to dataset root directory; others paths are relative to root
    # fmt: on


# [Reproduction] LLaVa-v15 (exact dataset used in all public LLaVa-v15 models)
@dataclass
class LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_mix665k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("data")


# [Multimodal-Only] LLava-v15 WITHOUT the Language-Only ShareGPT Data (No Co-Training)
@dataclass
class LLaVa_Multimodal_Only_Config(DatasetConfig):
    dataset_id: str = "llava-multimodal"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_stripped625k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V
@dataclass
class LLaVa_LVIS4V_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_mix888k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LRV-Instruct
@dataclass
class LLaVa_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lrv_mix1008k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# LLaVa-v15 + LVIS-Instruct-4V + LRV-Instruct
@dataclass
class LLaVa_LVIS4V_LRV_Config(DatasetConfig):
    dataset_id: str = "llava-lvis4v-lrv"

    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-v1.5-instruct/llava_v1_5_lvis4v_lrv_mix1231k.json"),
        Path("download/llava-v1.5-instruct/"),
    )
    dataset_root_dir: Path = Path("/mnt/fsx/skaramcheti/datasets/prismatic-vlms")


# CLEVR & Ablated CLEVR Dataset Configurations
@dataclass
class CLEVRConfig(DatasetConfig):
    dataset_id: str = "clevr"

    # Preprocessed CLEVR dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_CLEVR/clevr_train_qa_preprocessed.json"),
        Path("data/CLEVR_v1.0/images/"),
    )
    # Single-stage finetune
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_CLEVR/clevr_train_qa_preprocessed.json"),
        Path("data/CLEVR_v1.0/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class CLEVRValidationConfig(DatasetConfig):
    dataset_id: str = "clevr-validation"

    # Preprocessed CLEVR dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_CLEVR/clevr_val_qa_preprocessed.json"),
        Path("data/CLEVR_v1.0/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_CLEVR/clevr_val_qa_preprocessed.json"),
        Path("data/CLEVR_v1.0/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class CLEVRFrontConfig(DatasetConfig):
    dataset_id: str = "clevr-front"

    # Preprocessed CLEVR dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_CLEVR/clevr_train_qa_front_preprocessed.json"),
        Path("data/CLEVR_v1.0/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_CLEVR/clevr_train_qa_front_preprocessed.json"),
        Path("data/CLEVR_v1.0/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class CLEVRMixedConfig(DatasetConfig):
    dataset_id: str = "clevr-mixed"

    # Preprocessed CLEVR dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_CLEVR/clevr_train_qa_mixed_preprocessed.json"),
        Path("data/CLEVR_v1.0/images/"),
    )
    # We don't use finetune stage for CLEVR, but it's required for consistency
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/CLEVR_v1.0/questions/"),
        Path("data/CLEVR_v1.0/questions/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")


# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # === LLaVa v1.5 ===
    LLAVA_V15 = LLaVa_V15_Config

    LLAVA_MULTIMODAL_ONLY = LLaVa_Multimodal_Only_Config

    LLAVA_LVIS4V = LLaVa_LVIS4V_Config
    LLAVA_LRV = LLaVa_LRV_Config

    LLAVA_LVIS4V_LRV = LLaVa_LVIS4V_LRV_Config

    CLEVR = CLEVRConfig
    CLEVR_VALIDATION = CLEVRValidationConfig
    CLEVR_FRONT_ABLATED = CLEVRFrontConfig
    CLEVR_MIXED = CLEVRMixedConfig

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
