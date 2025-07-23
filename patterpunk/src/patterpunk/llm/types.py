"""
Type definitions for LLM tool calling functionality.

This module contains all TypedDict definitions and type aliases used across
the LLM components to avoid circular imports.
"""

from typing import List, TypedDict, Dict, Optional


class ToolFunctionParameters(TypedDict, total=False):
    """Parameters schema for a tool function."""

    type: str
    properties: Dict[str, Dict[str, str]]
    required: List[str]
    additionalProperties: bool


class ToolFunction(TypedDict):
    """Function definition within a tool."""

    name: str
    description: str
    parameters: ToolFunctionParameters
    strict: Optional[bool]


class Tool(TypedDict):
    """Tool definition for LLM function calling."""

    type: str  # Should be "function"
    function: ToolFunction


class ToolCallFunction(TypedDict):
    """Function call within a tool call response."""

    name: str
    arguments: str  # JSON string containing the arguments


class ToolCall(TypedDict):
    """Individual tool call from LLM response."""

    id: str
    type: str  # Should be "function"
    function: ToolCallFunction


# Type aliases for cleaner usage
ToolDefinition = List[Tool]
ToolCallList = List[ToolCall]
