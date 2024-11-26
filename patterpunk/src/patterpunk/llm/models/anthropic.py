from abc import ABC
from typing import List, Optional, Callable, get_args

from patterpunk.config import (
    anthropic,
    ANTHROPIC_DEFAULT_TEMPERATURE,
    ANTHROPIC_DEFAULT_TOP_P,
    ANTHROPIC_DEFAULT_TOP_K,
    ANTHROPIC_DEFAULT_MAX_TOKENS,
)
from patterpunk.llm.messages import (
    Message,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    AssistantMessage,
)
from patterpunk.llm.models.base import Model
from patterpunk.logger import logger


class AnthropicMaxTokensError(Exception):
    pass


class AnthropicNotImplemented(Exception):
    pass


class AnthropicModel(Model, ABC):
    def __init__(
        self,
        model: str,
        temperature: float = ANTHROPIC_DEFAULT_TEMPERATURE,
        top_p: float = ANTHROPIC_DEFAULT_TOP_P,
        top_k: int = ANTHROPIC_DEFAULT_TOP_K,
        max_tokens: int = ANTHROPIC_DEFAULT_MAX_TOKENS,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

    def generate_assistant_message(
        self, messages: List[Message], functions: Optional[List[Callable]] = None
    ) -> Message:
        system_prompt = "\n\n".join(
            [message.content for message in messages if message.role == ROLE_SYSTEM]
        )

        response = anthropic.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[
                message.to_dict()
                for message in messages
                if message.role in [ROLE_USER, ROLE_ASSISTANT]
                and not message.is_function_call
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if response.stop_reason in ["end_turn", "stop_sequence"]:
            content = "\n".join(
                [block.text for block in response.content if block.type == "text"]
            )
            return AssistantMessage(content)
        elif response.stop_reason == "max_tokens":
            raise AnthropicMaxTokensError("Model reached maximum tokens")
        elif response.stop_reason == "tool_use":
            raise AnthropicNotImplemented(
                "Tool use has not been implemented for Anthropic yet"
            )

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Ugh, this is very hacky, but anthropic doesn't have a get-all-models api and I don't
        want to manually maintain the list of models, so we're extracting it from the type
        hint in the python sdk.

        If anthropic ever offers a get-all-models api, we need to switch to it.
        """
        from anthropic.types import Model

        args = get_args(Model)

        model_definition = next((arg for arg in args if not isinstance(arg, type)))
        models = repr(model_definition).split("'")[1::2]

        models_snapshot_2024_08_22 = [
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]

        # let's perform some sanity checks to flag early on if new versions of the sdk broke this
        # hack. if we detect an issue, we'll warn about the mismatch and return a snapshot of
        # available models, so we don't blow up uses of this library just because anthropic
        # changed the python sdk.
        # After all, the user may actually be specifying models directly, so don't raise an
        # Exception here
        if not isinstance(models, list):
            logger.error("Anthropic models coud not be determined")
            return models_snapshot_2024_08_22
        if not all([isinstance(item, str) for item in models]):
            logger.error(
                "Not all items in the Anthropic models list were strings, something went wrong"
            )
            return models_snapshot_2024_08_22

        return models

    @staticmethod
    def get_name():
        return "Anthropic"
