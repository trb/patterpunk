from ..tool_types import ToolCallList
from .base import Message
from .roles import ROLE_TOOL_CALL


class ToolCallMessage(Message):
    
    def __init__(self, tool_calls: ToolCallList):
        super().__init__("", ROLE_TOOL_CALL)
        self.tool_calls = tool_calls

    def to_dict(self, prompt_for_structured_output: bool = False):
        return {"role": self.role, "tool_calls": self.tool_calls}

    def __repr__(self, truncate=True):
        tool_names = [
            call.get("function", {}).get("name", "unknown") for call in self.tool_calls
        ]
        return f'ToolCallMessage({", ".join(tool_names)})'