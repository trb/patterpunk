"""
All message-related exceptions with clear inheritance and purpose.

This module contains all exception classes used throughout the message system,
providing clear error handling and debugging capabilities.
"""


class BadParameterError(Exception):
    """Raised when message templating encounters unexpected parameters."""
    pass


class UnexpectedFunctionCallError(Exception):
    """Raised when an unexpected function call is encountered."""
    pass


class StructuredOutputNotPydanticLikeError(Exception):
    """Raised when structured output target is not a Pydantic-like model."""
    pass


class StructuredOutputFailedToParseError(Exception):
    """Raised when structured output parsing fails after extraction."""
    pass