"""
UserMessage with structured output and tool call control.

This module contains the UserMessage class for user inputs with support
for structured output parsing and tool call configuration.
"""

from typing import Union, List, Optional, Any

from ..cache import CacheChunk
from .base import Message
from .roles import ROLE_USER


class UserMessage(Message):
    """
    User message with structured output support and tool call control.
    
    UserMessage represents input from the user and supports:
    - Structured output parsing with Pydantic models
    - Control over whether tool calls are allowed for this message
    """
    
    def __init__(
        self, 
        content: Union[str, List[CacheChunk]], 
        structured_output: Optional[Any] = None, 
        allow_tool_calls: bool = True
    ):
        super().__init__(content, ROLE_USER)
        self.structured_output = structured_output
        self.allow_tool_calls = allow_tool_calls