import warnings
from typing import List, Optional

from ..tool_types import ToolCall, ToolCallList
from .base import Message
from .roles import ROLE_TOOL_CALL


class ToolCallMessage(Message):
    """
    Message type for tool/function calls made by the model.

    Contains a list of ToolCall dataclasses and optional thinking blocks
    that preceded the tool calls (for models with extended thinking).
    """

    def __init__(
        self,
        tool_calls: ToolCallList,
        thinking_blocks: Optional[List[dict]] = None,
        thinking_token_count: Optional[int] = None,
        id: Optional[str] = None,
    ):
        super().__init__("", ROLE_TOOL_CALL, id=id)
        self.tool_calls = tool_calls
        self.thinking_blocks = thinking_blocks or []
        self._thinking_token_count = thinking_token_count

    @property
    def thinking_token_count(self) -> Optional[int]:
        if self._thinking_token_count is None and self.thinking_blocks:
            warnings.warn(
                "thinking_token_count is not available — the provider does not "
                "report separate thinking token counts.",
                stacklevel=2,
            )
        return self._thinking_token_count

    @thinking_token_count.setter
    def thinking_token_count(self, value: Optional[int]):
        self._thinking_token_count = value

    def to_dict(self, prompt_for_structured_output: bool = False) -> dict:
        """
        Convert to dictionary format for API calls.

        Tool calls are converted to OpenAI-compatible format.
        Thinking blocks are included if present.
        For persistence, use serialize() instead.
        """
        result = {
            "role": self.role,
            "tool_calls": [tc.to_openai_format() for tc in self.tool_calls],
        }
        if self.thinking_blocks:
            result["thinking_blocks"] = self.thinking_blocks
        return result

    def serialize(self) -> dict:
        """
        Serialize to dict for persistence.

        Use this method for storing messages in databases. For API calls
        to LLM providers, use to_dict() instead.
        """
        result = {
            "type": "tool_call",
            "id": self.id,
            "tool_calls": [tc.to_openai_format() for tc in self.tool_calls],
        }
        if self.thinking_blocks:
            result["thinking_blocks"] = self.thinking_blocks
        if self._thinking_token_count is not None:
            result["thinking_token_count"] = self._thinking_token_count
        return result

    @classmethod
    def deserialize(cls, data: dict) -> "ToolCallMessage":
        """Deserialize from dict."""
        tool_calls = [ToolCall.from_openai_format(tc) for tc in data["tool_calls"]]
        return cls(
            tool_calls=tool_calls,
            thinking_blocks=data.get("thinking_blocks"),
            thinking_token_count=data.get("thinking_token_count"),
            id=data.get("id"),
        )

    def __repr__(self, truncate=True):
        tool_names = [tc.name for tc in self.tool_calls]
        return f'ToolCallMessage({", ".join(tool_names)})'
