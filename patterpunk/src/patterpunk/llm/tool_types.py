from dataclasses import dataclass
from typing import List, TypedDict, Dict, Optional, Literal, Union


# =============================================================================
# Tool Definition Types (for defining available tools)
# =============================================================================


class ToolFunctionParameters(TypedDict, total=False):
    type: str
    properties: Dict[str, Dict[str, str]]
    required: List[str]
    additionalProperties: bool


class ToolFunction(TypedDict):
    name: str
    description: str
    parameters: ToolFunctionParameters
    strict: Optional[bool]


class Tool(TypedDict):
    type: str
    function: ToolFunction


ToolDefinition = List[Tool]


# =============================================================================
# Tool Call Types (for tool calls made by the model)
# =============================================================================


# Legacy TypedDict format (for API boundary typing)
class ToolCallFunctionDict(TypedDict):
    name: str
    arguments: str


class ToolCallDict(TypedDict):
    id: str
    type: str
    function: ToolCallFunctionDict


@dataclass
class ToolCall:
    """
    Represents a tool call made by the model.

    This is the internal representation used throughout patterpunk.
    Use to_openai_format() when serializing for provider APIs.

    Attributes:
        id: Unique identifier for this tool call (used to match with results)
        name: Name of the function/tool being called
        arguments: JSON string containing the function arguments
        type: Type of tool call (always "function" for now)
    """

    id: str
    name: str
    arguments: str
    type: str = "function"

    def to_openai_format(self) -> ToolCallDict:
        """
        Convert to OpenAI-compatible dict format.

        This format is used by OpenAI, Anthropic, Bedrock, and most providers.
        """
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }

    @classmethod
    def from_openai_format(cls, d: ToolCallDict) -> "ToolCall":
        """
        Create a ToolCall from OpenAI-compatible dict format.

        Args:
            d: Dictionary with id, type, and function.name/function.arguments
        """
        return cls(
            id=d["id"],
            name=d["function"]["name"],
            arguments=d["function"]["arguments"],
            type=d.get("type", "function"),
        )


ToolCallList = List[ToolCall]


# =============================================================================
# Thinking Block Types (for model reasoning/thinking content)
# =============================================================================


@dataclass
class ThinkingBlock:
    """
    Represents a thinking/reasoning block from the model.

    Anthropic models with extended thinking enabled produce these blocks
    containing the model's reasoning process.

    Two types exist:
    - "thinking": Contains visible reasoning text and optional signature
    - "redacted_thinking": Contains encrypted data (safety-redacted content)

    Attributes:
        type: Either "thinking" or "redacted_thinking"
        thinking: The reasoning text (only for type="thinking")
        signature: Cryptographic signature for verification (only for type="thinking")
        data: Encrypted data (only for type="redacted_thinking")
    """

    type: Literal["thinking", "redacted_thinking"]
    thinking: Optional[str] = None
    signature: Optional[str] = None
    data: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dict format for API serialization."""
        if self.type == "thinking":
            result: dict = {"type": "thinking", "thinking": self.thinking}
            if self.signature:
                result["signature"] = self.signature
            return result
        else:
            return {"type": "redacted_thinking", "data": self.data}

    @classmethod
    def from_dict(cls, d: dict) -> "ThinkingBlock":
        """Create a ThinkingBlock from dict format."""
        return cls(
            type=d["type"],
            thinking=d.get("thinking"),
            signature=d.get("signature"),
            data=d.get("data"),
        )


ThinkingBlockList = List[ThinkingBlock]


# =============================================================================
# Utility Functions
# =============================================================================


def thinking_blocks_to_dicts(
    blocks: Optional[ThinkingBlockList],
) -> Optional[List[dict]]:
    """Convert a list of ThinkingBlocks to list of dicts for API serialization."""
    if blocks is None:
        return None
    return [block.to_dict() for block in blocks]


def thinking_blocks_from_dicts(
    dicts: Optional[List[dict]],
) -> Optional[ThinkingBlockList]:
    """Convert a list of dicts to list of ThinkingBlocks."""
    if dicts is None:
        return None
    return [ThinkingBlock.from_dict(d) for d in dicts]
