"""
base_prompter.py

Abstract class definition of a multi-turn prompt builder for ensuring consistent formatting for chat-based LLMs.
"""

from abc import ABC, abstractmethod
from typing import Optional

# Default System Prompt for Base Models
SYS_PROMPTS = {
    "prismatic": (
        # "You are a helpful language and vision assistant. "
        # "You are able to understand the visual content that the user provides, "
        # "and assist the user with a variety of tasks using natural language. "
        # (txu) =>> added specific prompts for spatial reasoning tasks
        # "By `behind` I mean further from the viewer. "
        # "By `in front of` I mean closer to the viewer. "
        # "By `left` I mean closer to the left edge of the image. "
        # "By `right` I mean closer to the right edge of the image. "
        # "Only answer with one word: `yes` or `no`. "
    ),
}

def format_system_prompt(system_prompt: str) -> str:
    return f"{system_prompt.strip()}\n\n"

class PromptBuilder(ABC):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        self.model_family = model_family

        # Only some models define a system prompt => let subclasses handle this logic!
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str: ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> None: ...

    @abstractmethod
    def get_prompt(self) -> str: ...


class PurePromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)
        # print(self.model_family)
        # self.system_prompt = format_system_prompt(
        #     SYS_PROMPTS[self.model_family] if system_prompt is None else system_prompt
        # )
        self.system_prompt = ""

        # TODO (siddk) =>> Can't always assume LlamaTokenizer --> FIX ME!
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"In: {msg}\nOut: "
        # self.wrap_human = lambda msg: f"{msg}"
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        # Special Handling for "system" prompt (turn_count == 0)
        if self.turn_count == 0:
            sys_message = self.wrap_human(self.system_prompt + message) # (txu) =>> what is wrap_human?
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

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> (if exists) because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()
