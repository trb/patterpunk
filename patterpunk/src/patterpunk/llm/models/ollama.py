from abc import ABC
from typing import List, Optional, Callable

from patterpunk.config import ollama
from patterpunk.llm.messages import Message, AssistantMessage
from patterpunk.llm.models.base import Model


class OllamaModel(Model, ABC):
    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        num_ctx: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens

    def generate_assistant_message(
        self, messages: List[Message], functions: Optional[List[Callable]] = None
    ) -> Message:
        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature

        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.presence_penalty is not None:
            options["repeat_penalty"] = self.presence_penalty
        if self.seed is not None:
            options["seed"] = self.seed
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens

        response = ollama.chat(
            model=self.model,
            messages=[
                message.to_dict()
                for message in messages
                if not message.is_function_call
            ],
            stream=False,
            options=options,
        )

        return AssistantMessage(response["message"]["content"])

    @staticmethod
    def get_name():
        return "Ollama"

    @staticmethod
    def get_available_models() -> List[str]:
        return [model["model"] for model in ollama.list()["models"]]
