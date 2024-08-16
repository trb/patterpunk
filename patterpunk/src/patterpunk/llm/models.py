from typing import List

from patterpunk.llm.messages import Message


class ModelNotImplemented(Exception):
    pass

class Model:
    def generate_assistant_message(self, messages: List[Message], functions: list | None = None) -> Message:
        raise ModelNotImplemented('You need to use a LLM-specific model')