"""
Type definitions for LLM functionality - refactored for modularity.

This module now re-exports types from focused modules to maintain backward compatibility
while providing better organization and maintainability.
"""

# Re-export tool calling types for backward compatibility
from .tool_types import (
    ToolFunctionParameters,
    ToolFunction,
    Tool,
    ToolCallFunction,
    ToolCall,
    ToolDefinition,
    ToolCallList,
)

# Re-export cache functionality for backward compatibility
from .cache import CacheChunk

# Re-export multimodal functionality
from .multimodal import MultimodalChunk

# Content can be a string, or a list of chunks (text/cache/multimodal)
from typing import Union, List
ContentType = Union[str, List[Union[CacheChunk, MultimodalChunk]]]
