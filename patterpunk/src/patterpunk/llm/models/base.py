from abc import ABC, abstractmethod
from typing import List, Callable, Optional

from patterpunk.llm.messages import Message


class ModelNotImplemented(Exception):
    pass


class Model(ABC):
    @abstractmethod
    def generate_assistant_message(
        self, messages: List[Message], functions: Optional[List[Callable]] = None
    ) -> Message:
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
