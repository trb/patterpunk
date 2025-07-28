"""
SystemMessage with system-specific behavior.

This module contains the SystemMessage class for system-level instructions
and prompts in LLM conversations.
"""

from typing import Union, List

from ..cache import CacheChunk
from .base import Message
from .roles import ROLE_SYSTEM


class SystemMessage(Message):
    """
    System message for providing instructions and context to the LLM.
    
    System messages are typically used to set the behavior, persona, or
    provide context that guides the LLM's responses throughout the conversation.
    """
    
    def __init__(self, content: Union[str, List[CacheChunk]]):
        super().__init__(content, ROLE_SYSTEM)