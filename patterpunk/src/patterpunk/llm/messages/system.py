from typing import Optional, Union, List

from ..chunks import CacheChunk
from .base import Message
from .roles import ROLE_SYSTEM
from .serialization import serialize_content, deserialize_content


class SystemMessage(Message):

    def __init__(self, content: Union[str, List[CacheChunk]], id: Optional[str] = None):
        super().__init__(content, ROLE_SYSTEM, id=id)

    def serialize(self) -> dict:
        """
        Serialize to dict for persistence.

        Use this method for storing messages in databases. For API calls
        to LLM providers, use the inherited to_dict() method instead.
        """
        return {
            "type": "system",
            "id": self.id,
            "content": serialize_content(self.content),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SystemMessage":
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
        return cls(content=content, id=data.get("id"))
