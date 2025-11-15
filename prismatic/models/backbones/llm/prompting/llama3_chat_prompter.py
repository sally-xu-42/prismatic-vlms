"""
llama3_prompter.py

Defines a PromptBuilder for building LLaMa-3 Chat Prompts --> not sure if this is "optimal", but this is the pattern
that's used by HF and other online tutorials.

Reference: https://huggingface.co/blog/llama3#how-to-prompt-llama-3
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder

# Default System Prompt for Prismatic Models
SYS_PROMPTS = {
    "prismatic": (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    ),
}


class LLaMa3ChatPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        # LLaMa-3 Specific
        self.bos = "<|begin_of_text|>"

        # Get role-specific "wrap" functions
        self.wrap_sys = lambda msg: f"<|start_header_id|>system<|end_header_id|> {msg.strip()} <|eot_id|>"
        self.wrap_human = lambda msg: f"<|start_header_id|>user<|end_header_id|> {msg} <|eot_id|>"
        self.wrap_gpt = lambda msg: f"<|start_header_id|>assistant<|end_header_id|> {msg} <|eot_id|>"

        # === `self.system_prompt` gets formatted once at init time ===
        self.system_prompt = self.wrap_sys(
            SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
        )

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.system_prompt + self.wrap_human(message)
            wrapped_message = sys_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.system_prompt + self.wrap_human(message)
            prompt_copy += sys_message

        else:
            human_message = self.wrap_human(message)
            prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()
