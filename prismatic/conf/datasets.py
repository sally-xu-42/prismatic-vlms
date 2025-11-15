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
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics/prismatic-vlms/data")


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
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics/prismatic-vlms/data")


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
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics/prismatic-vlms/data")


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
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics/prismatic-vlms/data")


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
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics/prismatic-vlms/data")


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
        # Path("data/preprocessed_CLEVR/clevr_train_qa_preprocessed_filtered.json"),
        Path("data/CLEVR_v1.0/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class CLEVRMiniConfig(DatasetConfig):
    dataset_id: str = "clevr-mini"

    # Preprocessed CLEVR dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_CLEVR/clevr_train_qa_mini_preprocessed.json"),
        Path("data/CLEVR_v1.0/images/"),
    )
    # Single-stage finetune
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_CLEVR/clevr_train_qa_mini_preprocessed.json"),
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

# ============ Basic THINGS Dataset Configurations ============
@dataclass
class THINGSConfig(DatasetConfig):
    dataset_id: str = "things"

    # Preprocessed THINGS dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class THINGSHypConfig(DatasetConfig):
    dataset_id: str = "things-hyp"

    # Preprocessed THINGS dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_hyp.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_hyp.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class THINGSAndHypConfig(DatasetConfig):
    dataset_id: str = "things+hyp"

    # Preprocessed THINGS dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_things+hyp.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_things+hyp.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class THINGSAbl10Config(DatasetConfig):
    dataset_id: str = "things+hyp-abl10"

    # Preprocessed THINGS dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_combined_ablated_10pct.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_combined_ablated_10pct.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class THINGSAbl30Config(DatasetConfig):
    dataset_id: str = "things+hyp-abl30"

    # Preprocessed THINGS dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_combined_ablated_30pct.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_combined_ablated_30pct.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class THINGSAbl50Config(DatasetConfig):
    dataset_id: str = "things+hyp-abl50"

    # Preprocessed THINGS dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_combined_ablated_50pct.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_combined_ablated_50pct.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class THINGSAbl70Config(DatasetConfig):
    dataset_id: str = "things+hyp-abl70"

    # Preprocessed THINGS dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_combined_ablated_70pct.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_combined_ablated_70pct.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

@dataclass
class THINGSAbl90Config(DatasetConfig):
    dataset_id: str = "things+hyp-abl90"

    # Preprocessed THINGS dataset json file
    align_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_combined_ablated_90pct.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/preprocessed_THINGS/train_combined_ablated_90pct.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

# ============ Baseline Configurations ============
# 1. LLaVa-v15: llava-v15
# 2. THINGS + LLaVa-v15 (Exact same dataset as used in all public LLaVa-v15 models, but with THINGS data added in)
@dataclass
class THINGS_LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "things+llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        Path("data/baseline/things+llava-v15_align.json"),
        Path("data/hypernymy_THINGS/images/"), # llava + things images are all in this directory, so we can point to it directly
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/baseline/things+llava-v15_finetune.json"),
        Path("data/hypernymy_THINGS/images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

# 3. THINGS + THINGS-HYP + LLaVa-v15 (Exact same dataset as used in all public LLaVa-v15 models, but with THINGS data added in)
@dataclass
class THINGS_HYP_LLaVa_V15_Config(DatasetConfig):
    dataset_id: str = "things+hyp+llava-v15"

    align_stage_components: Tuple[Path, Path] = (
        Path("data/baseline/things+hyp+llava-v15_align.json"),
        Path("data/all_images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("data/baseline/things+hyp+llava-v15_finetune.json"),
        Path("data/all_images/"),
    )
    dataset_root_dir: Path = Path("/share/data/speech/txu/vlm_semantics")

# ============ Ablation Configurations ============

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
    CLEVR_MINI = CLEVRMiniConfig

    THINGS = THINGSConfig
    THINGS_HYP = THINGSHypConfig
    THINGSAndHYP = THINGSAndHypConfig
    THINGS_ABL_10 = THINGSAbl10Config
    THINGS_ABL_30 = THINGSAbl30Config
    THINGS_ABL_50 = THINGSAbl50Config
    THINGS_ABL_70 = THINGSAbl70Config
    THINGS_ABL_90 = THINGSAbl90Config
    THINGS_LLAVA_V15 = THINGS_LLaVa_V15_Config 
    THINGS_HYP_LLAVA_V15 = THINGS_HYP_LLaVa_V15_Config 

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
