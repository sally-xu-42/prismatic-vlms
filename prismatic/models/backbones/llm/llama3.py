"""
llama3.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from peft import LoraConfig

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    LLaMa3ChatPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
)

# Registry =>> Support LLaMa-3 Models (from HF Transformers)
# fmt: off
LLAMA3_MODELS = {
    # === Pure Meta LLaMa-3 (non-instruct/chat-tuned) Models ===
    "llama-3-1b": {
        "llm_family": "llama3", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-3.2-1B"
    },
    # === Meta LLaMa-3 Chat Models ===
    "llama-3-1b-chat": {
        "llm_family": "llama3", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-3.2-1B-Instruct"
    },
}
# fmt: on


class LLaMa3LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = False,
        enable_peft = True, #      <<<<<<<<<<<<<<<         ################### set this to True for finetune stage #######################
        lora_config = LoraConfig(
            r=8, # <-- Changed from 128 to 8 for faster training
            lora_alpha=16, # <-- Should be 2x the rank (8 * 2 = 16)
            lora_dropout=0.0, # <-- Keep dropout at 0
            bias="none",
            target_modules= [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj"
            ],  # (txu =>>) "q_proj","k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj"
            task_type="CAUSAL_LM"
            ),
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            # enable_peft = enable_peft,
            # lora_peft_config = lora_config,
            **LLAMA3_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-3 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("llama-3") and not self.identifier.endswith("-chat"):
            return PurePromptBuilder

        elif self.identifier.startswith("llama-3") and self.identifier.endswith("-chat"):
            return LLaMa3ChatPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-3 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama3."""
        return torch.bfloat16
