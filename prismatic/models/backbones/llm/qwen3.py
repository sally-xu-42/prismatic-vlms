"""
qwen.py

Class definition for all LLMs derived from QwenForCausalLM.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from peft import LoraConfig

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import Qwen3PromptBuilder, PromptBuilder

# Registry ==> Support Qwen Models (from HF Transformers)
# [Attention] QWen3 requires transformers >= 4.51.0
# fmt: off
QWEN_MODELS = {
    # === Qwen-3-0.6B ===
    "qwen-3-0.6b": {
        "llm_family": "qwen", "llm_cls": Qwen3ForCausalLM, "hf_hub_path": "Qwen/Qwen3-0.6B"
    },
    # === Qwen-3-1.7B ===
    "qwen-3-1.7b": {
        "llm_family": "qwen", "llm_cls": Qwen3ForCausalLM, "hf_hub_path": "Qwen/Qwen3-1.7B"
    },
    # === Qwen-3-4B-Instruct ===
    "qwen-3-4b-inst": {
        "llm_family": "qwen", "llm_cls": Qwen3ForCausalLM, "hf_hub_path": "Qwen/Qwen3-4B-Instruct-2507"
    },
}
# fmt: on


class Qwen3LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 32768,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        # use_flash_attention_2: bool = True,
        # enable_peft = True, #      <<<<<<<<<<<<<<<         ################### set this to True for finetune stage #######################
        # lora_config = LoraConfig(
        #     r=8, # <-- Changed from 128 to 8 for faster training
        #     lora_alpha=16, # <-- Should be 2x the rank (8 * 2 = 16)
        #     lora_dropout=0.0, # <-- Keep dropout at 0
        #     bias="none",
        #     target_modules= [
        #         "q_proj",
        #         "k_proj",
        #         "v_proj",
        #         "o_proj"
        #     ],  # (txu =>>) "q_proj","k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj"
        #     task_type="CAUSAL_LM"
        #     ),
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            # use_flash_attention_2=use_flash_attention_2,
            # enable_peft = enable_peft,
            # lora_peft_config = lora_config,
            **QWEN_MODELS[llm_backbone_id],
        )

        # [Special Case] Qwen BOS and PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"bos_token": "<BOS>"})
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.bos_token_id = self.tokenizer.bos_token_id
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("qwen-3"):
            return Qwen3PromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen3DecoderLayer
    
    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
