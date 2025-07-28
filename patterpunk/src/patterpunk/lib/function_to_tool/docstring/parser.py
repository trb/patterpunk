"""
Main docstring parsing coordination and fallback handling.

This module orchestrates between advanced and simple parsing approaches,
implementing fallback logic and providing a unified interface.
"""

from typing import Callable, Tuple, Dict

from .advanced import parse_with_docstring_parser, is_advanced_parser_available
from .simple import parse_with_regex


def parse_function_docs(func: Callable) -> Tuple[str, Dict[str, str]]:
    """
    Parse function documentation with automatic fallback between parsing strategies.
    
    Attempts advanced parsing first (if available), then falls back to regex-based
    parsing if needed. Provides robust docstring processing across different
    documentation formats and library availability scenarios.
    
    :param func: Function to parse docstring from
    :return: Tuple of (description, parameter_descriptions)
    """
    if not func.__doc__:
        return func.__name__, {}

    # Try advanced parser first if available
    if is_advanced_parser_available():
        try:
            return parse_with_docstring_parser(func)
        except (ImportError, ValueError):
            # Fall back to simple parser if advanced parsing fails
            pass
    
    # Use simple regex-based parser as fallback
    return parse_with_regex(func)