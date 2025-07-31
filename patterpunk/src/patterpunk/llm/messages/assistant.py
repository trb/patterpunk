from typing import Optional, Any

from .base import Message
from .roles import ROLE_ASSISTANT


class AssistantMessage(Message):

    def __init__(
        self,
        content: str,
        structured_output: Optional[Any] = None,
        parsed_output: Optional[Any] = None,
    ):
        super().__init__(content, ROLE_ASSISTANT)
        self.structured_output = structured_output
        self._parsed_output = parsed_output
