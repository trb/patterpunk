from typing import Optional

from .base import Message
from .roles import ROLE_TOOL_RESULT


class ToolResultMessage(Message):
    """
    Message type for tool execution results.

    Links tool execution results back to original tool call requests via call_id.
    Different providers require different metadata:
    - OpenAI/Anthropic/Bedrock: Require call_id for linkage
    - Google Vertex AI: Requires function_name (no call_id)
    - Both fields optional: Providers validate what they need at serialization time

    Args:
        content: The tool execution result as a string
        call_id: Optional ID linking to original ToolCallMessage (OpenAI/Anthropic/Bedrock)
        function_name: Optional name of executed function (Google/general use)
        is_error: Whether the tool execution failed (Anthropic error handling)
    """

    def __init__(
        self,
        content: str,
        call_id: Optional[str] = None,
        function_name: Optional[str] = None,
        is_error: bool = False,
    ):
        super().__init__(content, ROLE_TOOL_RESULT)
        self.call_id = call_id
        self.function_name = function_name
        self.is_error = is_error

    def to_dict(self, prompt_for_structured_output: bool = False):
        """
        Basic dictionary representation (not provider-specific).
        Provider models handle their own serialization format.
        """
        result = {
            "role": self.role,
            "content": self.content,
        }

        if self.call_id is not None:
            result["call_id"] = self.call_id

        if self.function_name is not None:
            result["function_name"] = self.function_name

        if self.is_error:
            result["is_error"] = self.is_error

        return result

    def __repr__(self, truncate=True):
        """
        Debug representation showing linkage metadata and truncated content.
        """
        content_str = str(self.content)
        truncated_content = (
            content_str
            if len(content_str) < 30 or not truncate
            else f"{content_str[:30]}..."
        )

        metadata_parts = []
        if self.call_id:
            metadata_parts.append(f"call_id={self.call_id}")
        if self.function_name:
            metadata_parts.append(f"function={self.function_name}")
        if self.is_error:
            metadata_parts.append("is_error=True")

        metadata_str = ", ".join(metadata_parts) if metadata_parts else "no_linkage"

        return f'ToolResultMessage({metadata_str}, content="{truncated_content}")'
