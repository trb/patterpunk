"""
Serialization utilities for message persistence.

This module provides utilities for serializing and deserializing messages
to enable storing conversations in databases and resuming them later.
"""

import importlib
import json
from typing import Any, List, Optional, Union

from patterpunk.lib.structured_output import has_model_schema, get_model_schema
from patterpunk.llm.chunks import CacheChunk, MultimodalChunk, TextChunk
from patterpunk.llm.types import ContentType


class DynamicStructuredOutput:
    """
    Fallback for structured_output when original Pydantic model is unavailable.

    This class satisfies patterpunk's structured_output interface but returns
    dicts instead of typed models. It is used when the original Pydantic model
    class cannot be imported during deserialization.

    Note: This class does not perform schema validation - it only parses JSON.
    For full validation, ensure the original Pydantic model is importable.
    """

    def __init__(self, schema: dict):
        self._schema = schema

    def model_json_schema(self) -> dict:
        """Return the stored JSON schema."""
        return self._schema

    def model_validate_json(self, json_str: str) -> dict:
        """Parse JSON string and return as dict (no schema validation)."""
        return json.loads(json_str)

    def model_validate(self, data: dict) -> dict:
        """Return data as-is (no schema validation)."""
        return data


def serialize_content(content: ContentType) -> dict:
    """
    Serialize message content (string or chunk list) to dict.

    Args:
        content: Either a string or a list of TextChunk/CacheChunk/MultimodalChunk

    Returns:
        Dict with 'type' field ('string' or 'chunks') and content data
    """
    if isinstance(content, str):
        return {"type": "string", "value": content}

    return {"type": "chunks", "chunks": [chunk.serialize() for chunk in content]}


def deserialize_content(data: dict) -> ContentType:
    """
    Deserialize content from dict.

    Args:
        data: Dict with 'type' field and content data

    Returns:
        Either a string or a list of chunk objects
    """
    if data["type"] == "string":
        return data["value"]

    chunks: List[Union[TextChunk, CacheChunk, MultimodalChunk]] = []
    for chunk_data in data["chunks"]:
        chunk_type = chunk_data["type"]
        if chunk_type == "text":
            chunks.append(TextChunk.from_dict(chunk_data))
        elif chunk_type == "cache":
            chunks.append(CacheChunk.from_dict(chunk_data))
        elif chunk_type == "multimodal":
            chunks.append(MultimodalChunk.from_dict(chunk_data))
        else:
            raise ValueError(f"Unknown chunk type: {chunk_type}")
    return chunks


def serialize_structured_output(structured_output: Any) -> Optional[dict]:
    """
    Serialize structured_output reference for persistence.

    Stores both the JSON schema and the class reference (module path + class name)
    to enable reconstruction during deserialization.

    Args:
        structured_output: A Pydantic model class or None

    Returns:
        Dict with 'schema' and/or 'class_ref' fields, or None
    """
    if structured_output is None:
        return None

    # Handle DynamicStructuredOutput (already has schema, no class_ref)
    if isinstance(structured_output, DynamicStructuredOutput):
        return {"schema": structured_output._schema}

    result = {}

    # Store schema if available
    if has_model_schema(structured_output):
        result["schema"] = get_model_schema(structured_output)

    # Store class reference for import attempt
    if hasattr(structured_output, "__module__") and hasattr(
        structured_output, "__name__"
    ):
        result["class_ref"] = (
            f"{structured_output.__module__}.{structured_output.__name__}"
        )

    return result if result else None


def deserialize_structured_output(data: Optional[dict]) -> Any:
    """
    Deserialize structured_output from stored data.

    Attempts to import the original Pydantic model class using the stored
    class reference. Falls back to DynamicStructuredOutput if import fails.

    Priority:
    1. Try dynamic import from class_ref
    2. Fall back to DynamicStructuredOutput (no validation, just parsing)

    Args:
        data: Dict with 'schema' and/or 'class_ref' fields, or None

    Returns:
        The original Pydantic model class, DynamicStructuredOutput, or None
    """
    if data is None:
        return None

    # Try dynamic import from stored class reference
    if "class_ref" in data:
        try:
            module_path, class_name = data["class_ref"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError, ValueError):
            pass  # Fall through to fallback

    # Fallback to dynamic wrapper (no validation, just JSON parsing)
    if "schema" in data:
        return DynamicStructuredOutput(data["schema"])

    return None


def serialize_message(message) -> dict:
    """
    Serialize any message type to dict for persistence.

    This is a convenience function that calls the appropriate serialize()
    method based on the message type.

    Args:
        message: A message instance (SystemMessage, UserMessage, etc.)

    Returns:
        Dict suitable for JSON serialization and database storage
    """
    return message.serialize()


def message_from_dict(data: dict) -> "Message":
    """
    Deserialize any message type from dict.

    This is a convenience function that dispatches to the appropriate
    message class based on the 'type' field (preferred) or 'role' field (fallback).

    For messages with structured_output, attempts dynamic import of the
    original Pydantic model class. Falls back to DynamicStructuredOutput
    if import fails.

    Args:
        data: Serialized message dict with 'type' or 'role' field

    Returns:
        The appropriate Message subclass instance

    Raises:
        ValueError: If the message type is unknown
    """
    # Import here to avoid circular imports
    from patterpunk.llm.messages.system import SystemMessage
    from patterpunk.llm.messages.user import UserMessage
    from patterpunk.llm.messages.assistant import AssistantMessage
    from patterpunk.llm.messages.tool_call import ToolCallMessage
    from patterpunk.llm.messages.tool_result import ToolResultMessage

    # Support both 'type' (new format) and 'role' (legacy format)
    msg_type = data.get("type") or data.get("role")

    if msg_type == "system":
        return SystemMessage.from_dict(data)
    elif msg_type == "user":
        return UserMessage.from_dict(data)
    elif msg_type == "assistant":
        return AssistantMessage.from_dict(data)
    elif msg_type in ("tool_call", "tool-call"):
        return ToolCallMessage.from_dict(data)
    elif msg_type in ("tool_result", "tool-result"):
        return ToolResultMessage.from_dict(data)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
