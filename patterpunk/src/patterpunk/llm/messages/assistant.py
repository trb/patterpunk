"""
AssistantMessage with response parsing capabilities.

This module contains the AssistantMessage class for LLM-generated responses
with support for structured output parsing and caching.
"""

from typing import Optional, Any

from .base import Message
from .roles import ROLE_ASSISTANT


class AssistantMessage(Message):
    """
    Assistant message representing LLM-generated responses.
    
    AssistantMessage handles responses from the LLM and supports:
    - Structured output parsing with result caching
    - Response content analysis and processing
    """
    
    def __init__(
        self, 
        content: str, 
        structured_output: Optional[Any] = None, 
        parsed_output: Optional[Any] = None
    ):
        super().__init__(content, ROLE_ASSISTANT)
        self.structured_output = structured_output
        self._parsed_output = parsed_output