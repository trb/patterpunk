"""
Docstring parsing sub-module with fallback logic.

This module provides unified docstring parsing with automatic fallback
between advanced and simple parsing strategies.
"""

# Export the main parsing function with fallback logic
from .parser import parse_function_docs

# Export individual parsing strategies for advanced use cases
from .advanced import parse_with_docstring_parser, is_advanced_parser_available
from .simple import parse_with_regex