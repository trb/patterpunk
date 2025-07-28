"""
Chat module exporting Chat class with preserved API.

This module provides the Chat class interface while using the new modular
structure underneath for better maintainability and testing.
"""

# Export the main Chat class for backward compatibility
from .core import Chat

# Export chat-specific exceptions
from .exceptions import StructuredOutputParsingError

# Export utility functions for internal use
from .tools import configure_tools, configure_mcp_servers, execute_mcp_tool_calls
from .structured_output import get_parsed_output_with_retry