from abc import ABC, abstractmethod
from typing import List, Optional, Union

from patterpunk.llm.messages import Message
from patterpunk.llm.types import ToolDefinition


class ModelNotImplemented(Exception):
    pass


class Model(ABC):
    @abstractmethod
    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
    ) -> Union[Message, "ToolCallMessage"]:
        raise ModelNotImplemented("You need to use a LLM-specific model")

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise ModelNotImplemented("Models need to implement the get_name() method")

    @staticmethod
    @abstractmethod
    def get_available_models() -> List[str]:
        raise ModelNotImplemented(
            "Models need to implement the get_available_models() method"
        )
