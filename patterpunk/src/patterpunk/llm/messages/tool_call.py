"""
ToolCallMessage with tool execution results.

This module contains the ToolCallMessage class for representing tool calls
made by the LLM and their execution context.
"""

from ..tool_types import ToolCallList
from .base import Message
from .roles import ROLE_TOOL_CALL


class ToolCallMessage(Message):
    """
    Tool call message representing LLM-initiated function calls.
    
    ToolCallMessage handles tool calls from the LLM, containing the
    function names, arguments, and call IDs for tool execution.
    """
    
    def __init__(self, tool_calls: ToolCallList):
        """
        Represents a tool call message from the LLM.

        :param tool_calls: List of tool calls, each containing id, function name, and arguments
        """
        super().__init__("", ROLE_TOOL_CALL)
        self.tool_calls = tool_calls

    def to_dict(self, prompt_for_structured_output: bool = False):
        """Convert to dictionary format with tool_calls instead of content."""
        return {"role": self.role, "tool_calls": self.tool_calls}

    def __repr__(self, truncate=True):
        """String representation showing tool names being called."""
        tool_names = [
            call.get("function", {}).get("name", "unknown") for call in self.tool_calls
        ]
        return f'ToolCallMessage({", ".join(tool_names)})'