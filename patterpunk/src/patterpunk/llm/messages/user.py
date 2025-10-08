from typing import Union, List, Optional, Any

from ..chunks import CacheChunk
from ..types import ContentType
from .base import Message
from .roles import ROLE_USER


class UserMessage(Message):

    def __init__(
        self,
        content: ContentType,
        structured_output: Optional[Any] = None,
        allow_tool_calls: bool = True,
    ):
        super().__init__(content, ROLE_USER)
        self.structured_output = structured_output
        self.allow_tool_calls = allow_tool_calls
