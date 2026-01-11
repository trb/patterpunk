"""
Type definitions for LLM functionality - refactored for modularity.

This module now re-exports types from focused modules to maintain backward compatibility
while providing better organization and maintainability.
"""

from .tool_types import (
    # Tool definition types
    ToolFunctionParameters,
    ToolFunction,
    Tool,
    ToolDefinition,
    # Tool call types (dataclass)
    ToolCall,
    ToolCallList,
    # Tool call types (TypedDict - for API boundary typing)
    ToolCallFunctionDict,
    ToolCallDict,
    # Thinking block types
    ThinkingBlock,
    ThinkingBlockList,
    thinking_blocks_to_dicts,
    thinking_blocks_from_dicts,
)

# Backward compatibility alias
ToolCallFunction = ToolCallFunctionDict

from .chunks import CacheChunk, MultimodalChunk, TextChunk

from typing import Union, List

ContentType = Union[str, List[Union[TextChunk, CacheChunk, MultimodalChunk]]]
