"""
Chat-specific exceptions and error handling.

This module contains exception classes specifically related to chat functionality,
providing clear error types for chat operations and orchestration.
"""


class StructuredOutputParsingError(Exception):
    """Raised when chat-level structured output parsing fails after retry attempts."""
    pass