"""
Messages module with clean public API re-exports.

This module provides a unified interface to all message types and utilities
while maintaining the new modular structure underneath.
"""

# Re-export role constants for backward compatibility
from .roles import (
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_TOOL_CALL,
)

# Re-export all message exceptions for backward compatibility
from .exceptions import (
    BadParameterError,
    UnexpectedFunctionCallError,
    StructuredOutputNotPydanticLikeError,
    StructuredOutputFailedToParseError,
)

# Re-export core Message class for backward compatibility
from .base import Message

# Re-export all message types for backward compatibility
from .system import SystemMessage
from .user import UserMessage
from .assistant import AssistantMessage
from .tool_call import ToolCallMessage

# Re-export utility functions for internal use
from .templating import format_content
from .cache import get_content_as_string, has_cacheable_content, get_cache_chunks, has_multimodal_content, get_multimodal_chunks
from .structured_output import parse_structured_output