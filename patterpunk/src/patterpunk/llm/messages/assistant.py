from typing import Optional, Any, Union, List

from ..chunks import CacheChunk, MultimodalChunk, TextChunk
from ..types import ContentType
from .base import Message
from .roles import ROLE_ASSISTANT
from .cache import get_content_as_string
from .serialization import (
    serialize_content,
    deserialize_content,
    serialize_structured_output,
    deserialize_structured_output,
)


class AssistantMessage(Message):

    def __init__(
        self,
        content: ContentType,
        structured_output: Optional[Any] = None,
        parsed_output: Optional[Any] = None,
        thinking_blocks: Optional[List[dict]] = None,
    ):
        super().__init__(content, ROLE_ASSISTANT)
        self.structured_output = structured_output
        self._parsed_output = parsed_output
        self._raw_content = content
        self.thinking_blocks = thinking_blocks or []

    @property
    def content(self) -> str:
        return get_content_as_string(self._raw_content)

    @content.setter
    def content(self, value: ContentType) -> None:
        self._raw_content = value

    @property
    def chunks(self) -> Optional[List[Union[TextChunk, CacheChunk, MultimodalChunk]]]:
        if isinstance(self._raw_content, str):
            return None
        elif isinstance(self._raw_content, list):
            return self._raw_content
        return None

    @property
    def texts(self) -> List[Union[TextChunk, CacheChunk]]:
        if isinstance(self._raw_content, str):
            return []
        elif isinstance(self._raw_content, list):
            return [
                chunk
                for chunk in self._raw_content
                if isinstance(chunk, (TextChunk, CacheChunk))
            ]
        return []

    @property
    def images(self) -> List[MultimodalChunk]:
        if isinstance(self._raw_content, str):
            return []
        elif isinstance(self._raw_content, list):
            return [
                chunk
                for chunk in self._raw_content
                if isinstance(chunk, MultimodalChunk)
                and chunk.media_type
                and chunk.media_type.startswith("image/")
            ]
        return []

    @property
    def videos(self) -> List[MultimodalChunk]:
        if isinstance(self._raw_content, str):
            return []
        elif isinstance(self._raw_content, list):
            return [
                chunk
                for chunk in self._raw_content
                if isinstance(chunk, MultimodalChunk)
                and chunk.media_type
                and chunk.media_type.startswith("video/")
            ]
        return []

    @property
    def audios(self) -> List[MultimodalChunk]:
        if isinstance(self._raw_content, str):
            return []
        elif isinstance(self._raw_content, list):
            return [
                chunk
                for chunk in self._raw_content
                if isinstance(chunk, MultimodalChunk)
                and chunk.media_type
                and chunk.media_type.startswith("audio/")
            ]
        return []

    @property
    def has_thinking(self) -> bool:
        """Check if this message contains thinking blocks."""
        return len(self.thinking_blocks) > 0

    @property
    def thinking_text(self) -> Optional[str]:
        """
        Get the combined thinking text from all thinking blocks.
        Returns None if there are no thinking blocks.
        Redacted thinking blocks are excluded as they contain encrypted data.
        """
        if not self.has_thinking:
            return None

        thinking_parts = []
        for block in self.thinking_blocks:
            if block.get("type") == "thinking" and "thinking" in block:
                thinking_parts.append(block["thinking"])

        return "\n".join(thinking_parts) if thinking_parts else None

    def to_dict(self, prompt_for_structured_output: bool = False) -> dict:
        """
        Convert to dictionary format for API calls.

        Includes thinking_blocks when present (for extended thinking models).
        For persistence, use serialize() instead.
        """
        result = super().to_dict(prompt_for_structured_output)
        if self.thinking_blocks:
            result["thinking_blocks"] = self.thinking_blocks
        return result

    def serialize(self) -> dict:
        """
        Serialize to dict for persistence.

        Use this method for storing messages in databases. For API calls
        to LLM providers, use to_dict() instead.

        Note: _parsed_output is derived from content, so it's not stored.
        """
        result = {
            "type": "assistant",
            "content": serialize_content(self._raw_content),
        }
        if self.thinking_blocks:
            result["thinking_blocks"] = self.thinking_blocks
        if self.structured_output:
            result["structured_output"] = serialize_structured_output(
                self.structured_output
            )
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "AssistantMessage":
        """
        Deserialize from dict.

        Handles both persistence format (content as dict) and API format
        (content as string) for flexibility.
        """
        raw_content = data["content"]
        if isinstance(raw_content, dict):
            content = deserialize_content(raw_content)
        else:
            content = raw_content
        return cls(
            content=content,
            structured_output=deserialize_structured_output(
                data.get("structured_output")
            ),
            thinking_blocks=data.get("thinking_blocks"),
        )
