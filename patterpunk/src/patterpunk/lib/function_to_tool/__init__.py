"""
Function-to-tool conversion module with modular architecture.

This module provides comprehensive function-to-tool conversion capabilities
through a well-organized pipeline of specialized components.
"""

# Export main conversion functions for backward compatibility
from .converter import function_to_tool, functions_to_tools

# Export specialized components for advanced use cases
from .inspection import analyze_function_signature, get_function_name
from .schema import create_pydantic_model_from_fields, generate_openai_compatible_schema
from .docstring import parse_function_docs
from .cleaning import clean_description