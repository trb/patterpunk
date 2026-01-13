from typing import Union, List, Optional, Any

from ..chunks import CacheChunk
from ..types import ContentType
from .base import Message
from .roles import ROLE_USER
from .serialization import (
    serialize_content,
    deserialize_content,
    serialize_structured_output,
    deserialize_structured_output,
)


class UserMessage(Message):

    def __init__(
        self,
        content: ContentType,
        structured_output: Optional[Any] = None,
        allow_tool_calls: bool = True,
        id: Optional[str] = None,
    ):
        super().__init__(content, ROLE_USER, id=id)
        self.structured_output = structured_output
        self.allow_tool_calls = allow_tool_calls

    def serialize(self) -> dict:
        """
        Serialize to dict for persistence.

        Use this method for storing messages in databases. For API calls
        to LLM providers, use the inherited to_dict() method instead.
        """
        result = {
            "type": "user",
            "id": self.id,
            "content": serialize_content(self.content),
            "allow_tool_calls": self.allow_tool_calls,
        }
        if self.structured_output:
            result["structured_output"] = serialize_structured_output(
                self.structured_output
            )
        return result

    @classmethod
    def deserialize(cls, data: dict) -> "UserMessage":
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
            allow_tool_calls=data.get("allow_tool_calls", True),
            id=data.get("id"),
        )
