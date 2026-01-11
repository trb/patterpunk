from typing import List, Optional

from ..tool_types import ToolCallList
from .base import Message
from .roles import ROLE_TOOL_CALL


class ToolCallMessage(Message):
    """
    Message type for tool/function calls made by the model.

    Contains a list of ToolCall dataclasses and optional thinking blocks
    that preceded the tool calls (for models with extended thinking).
    """

    def __init__(
        self, tool_calls: ToolCallList, thinking_blocks: Optional[List[dict]] = None
    ):
        super().__init__("", ROLE_TOOL_CALL)
        self.tool_calls = tool_calls
        self.thinking_blocks = thinking_blocks or []

    def to_dict(self, prompt_for_structured_output: bool = False):
        """
        Convert to dictionary format for serialization.

        Tool calls are converted to OpenAI-compatible format.
        Thinking blocks are included if present.
        """
        result = {
            "role": self.role,
            "tool_calls": [tc.to_openai_format() for tc in self.tool_calls],
        }
        if self.thinking_blocks:
            result["thinking_blocks"] = self.thinking_blocks
        return result

    def __repr__(self, truncate=True):
        tool_names = [tc.name for tc in self.tool_calls]
        return f'ToolCallMessage({", ".join(tool_names)})'
