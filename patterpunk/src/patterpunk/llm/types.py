"""
Type definitions for LLM tool calling functionality.

This module contains all TypedDict definitions and type aliases used across
the LLM components to avoid circular imports.
"""

from datetime import timedelta
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


class CacheChunk:
    """
    Represents a chunk of content that can be cached by LLM providers.
    
    This class allows fine-grained control over which parts of a message
    should be cached, enabling cost optimization across different providers.
    """
    
    def __init__(
        self, 
        content: str, 
        cacheable: bool = False, 
        ttl: Optional[timedelta] = None
    ):
        """
        Initialize a cache chunk.
        
        :param content: The text content of this chunk
        :param cacheable: Whether this chunk should be cached by the provider
        :param ttl: Time-to-live for the cache (provider-specific interpretation)
        """
        self.content = content
        self.cacheable = cacheable
        self.ttl = ttl
    
    def __repr__(self):
        cache_info = f", cacheable={self.cacheable}"
        if self.ttl:
            cache_info += f", ttl={self.ttl}"
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f'CacheChunk("{content_preview}"{cache_info})'


# Type aliases for cleaner usage
ToolDefinition = List[Tool]
ToolCallList = List[ToolCall]
