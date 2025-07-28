"""
Structured output parsing, retry logic, and validation for messages.

This module handles the complex logic for parsing structured output from message content,
including JSON extraction, Pydantic model validation, and error recovery mechanisms.
"""

from typing import Union, List, Optional, Any

from patterpunk.lib.extract_json import extract_json
from patterpunk.logger import logger
from ..cache import CacheChunk
from .exceptions import StructuredOutputNotPydanticLikeError, StructuredOutputFailedToParseError
from .cache import get_content_as_string


def parse_structured_output(
    content: Union[str, List[CacheChunk]], 
    structured_output: Any,
    role: str,
    cached_parsed_output: Optional[Any] = None
) -> Any:
    """
    Parse structured output from message content using Pydantic model validation.
    
    Attempts to extract JSON from content and validate it against the provided
    structured output model (should be a Pydantic model).
    
    :param content: Message content to parse
    :param structured_output: Pydantic model class for validation
    :param role: Message role for error reporting
    :param cached_parsed_output: Previously parsed output to return if available
    :return: Parsed and validated structured output object
    :raises StructuredOutputNotPydanticLikeError: If model doesn't support parsing
    :raises StructuredOutputFailedToParseError: If parsing fails after all attempts
    """
    if cached_parsed_output is not None:
        return cached_parsed_output

    if not structured_output:
        return None

    # Validate that the structured_output is Pydantic-like
    if not getattr(structured_output, "parse_raw", None) and not getattr(
        structured_output, "model_validate_json", None
    ):
        raise StructuredOutputNotPydanticLikeError(
            f"[MESSAGE][{role}] The provided structured_output is not a pydantic model (missing parse_raw or model_validate_json)"
        )

    # Extract JSON from content
    content_str = get_content_as_string(content)
    json_messages = extract_json(content_str)
    
    # Try to parse each extracted JSON
    for json_message in json_messages:
        try:
            if getattr(structured_output, "model_validate_json", None):
                obj = structured_output.model_validate_json(json_message)
            else:
                obj = structured_output.parse_raw(json_message)

            return obj
        except Exception as error:
            logger.debug(
                f"[MESSAGE][{role}][STRUCTURED_OUTPUT] Failed to parse response {json_message}: {error}",
                exc_info=error,
            )

    # If we get here, no JSON could be parsed successfully
    raise StructuredOutputFailedToParseError(
        f"[MESSAGE][{role}] Structured output could not be parsed, message: \n{content_str}"
    )