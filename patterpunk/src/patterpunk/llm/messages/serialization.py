"""
Serialization utilities for message persistence.

This module provides utilities for serializing and deserializing messages
to enable storing conversations in databases and resuming them later.
"""

import importlib
import json
from typing import Any, Optional

from patterpunk.lib.structured_output import has_model_schema, get_model_schema
from patterpunk.llm.chunks import CacheChunk, MultimodalChunk, TextChunk
from patterpunk.llm.types import ContentType
from patterpunk.logger import logger_llm


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

    Raises:
        ValueError: If required fields are missing or content type is unknown
    """
    content_type = data.get("type")
    if not content_type:
        raise ValueError("Content data missing required 'type' field")

    if content_type == "string":
        if "value" not in data:
            raise ValueError("String content missing required 'value' field")
        return data["value"]

    if content_type != "chunks":
        raise ValueError(f"Unknown content type: {content_type}")

    if "chunks" not in data:
        raise ValueError("Chunks content missing required 'chunks' field")

    chunks: list[TextChunk | CacheChunk | MultimodalChunk] = []
    for chunk_data in data["chunks"]:
        chunk_type = chunk_data.get("type")
        if chunk_type == "text":
            chunks.append(TextChunk.deserialize(chunk_data))
        elif chunk_type == "cache":
            chunks.append(CacheChunk.deserialize(chunk_data))
        elif chunk_type == "multimodal":
            chunks.append(MultimodalChunk.deserialize(chunk_data))
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
        class_ref = data["class_ref"]
        try:
            module_path, class_name = class_ref.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError, ValueError) as e:
            logger_llm.debug(
                f"Could not import structured_output class '{class_ref}', "
                f"falling back to DynamicStructuredOutput: {e}"
            )

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


def deserialize_message(data: dict):
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
        One of: SystemMessage, UserMessage, AssistantMessage,
        ToolCallMessage, or ToolResultMessage

    Raises:
        ValueError: If the message type is unknown
    """
    # Import here to avoid circular imports
    from patterpunk.llm.messages.system import SystemMessage
    from patterpunk.llm.messages.user import UserMessage
    from patterpunk.llm.messages.assistant import AssistantMessage
    from patterpunk.llm.messages.tool_call import ToolCallMessage
    from patterpunk.llm.messages.tool_result import ToolResultMessage

    deserializers = {
        "system": SystemMessage.deserialize,
        "user": UserMessage.deserialize,
        "assistant": AssistantMessage.deserialize,
        "tool_call": ToolCallMessage.deserialize,
        "tool-call": ToolCallMessage.deserialize,
        "tool_result": ToolResultMessage.deserialize,
        "tool-result": ToolResultMessage.deserialize,
    }

    # Support both 'type' (new format) and 'role' (legacy format)
    msg_type = data.get("type") or data.get("role")
    deserializer = deserializers.get(msg_type)

    if deserializer is None:
        raise ValueError(f"Unknown message type: {msg_type}")

    return deserializer(data)
