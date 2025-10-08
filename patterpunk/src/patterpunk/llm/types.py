"""
Type definitions for LLM functionality - refactored for modularity.

This module now re-exports types from focused modules to maintain backward compatibility
while providing better organization and maintainability.
"""

from .tool_types import (
    ToolFunctionParameters,
    ToolFunction,
    Tool,
    ToolCallFunction,
    ToolCall,
    ToolDefinition,
    ToolCallList,
)

from .chunks import CacheChunk, MultimodalChunk, TextChunk

from typing import Union, List

ContentType = Union[str, List[Union[TextChunk, CacheChunk, MultimodalChunk]]]
