"""
Main pipeline orchestration for function-to-tool conversion.

This module coordinates between all the specialized components to provide
the complete conversion pipeline with proper error handling and flow control.
"""

from typing import Callable, Dict, Any, List

from .inspection import analyze_function_signature, get_function_name
from .schema import create_pydantic_model_from_fields, generate_openai_compatible_schema
from .docstring import parse_function_docs
from .cleaning import clean_description


def function_to_tool(func: Callable) -> Dict[str, Any]:
    """
    Convert a Python function to OpenAI-compatible tool definition.
    
    This is the main entry point that orchestrates the entire conversion pipeline:
    1. Function signature analysis
    2. Pydantic model creation
    3. JSON schema generation
    4. Docstring parsing
    5. Description cleaning and integration
    6. Final tool definition assembly
    
    :param func: Python function to convert
    :return: OpenAI-compatible tool definition dictionary
    :raises ValueError: If function cannot be analyzed or converted
    """
    # Step 1: Analyze function signature and extract parameters
    signature, type_hints, fields = analyze_function_signature(func)
    function_name = get_function_name(func)
    
    # Step 2: Create Pydantic model and generate schema
    model = create_pydantic_model_from_fields(function_name, fields)
    schema = generate_openai_compatible_schema(model)
    
    # Step 3: Parse and clean function documentation
    description, param_descriptions = parse_function_docs(func)
    cleaned_description = clean_description(description)
    
    # Step 4: Integrate parameter descriptions into schema
    if param_descriptions and "properties" in schema:
        for param_name, param_schema in schema["properties"].items():
            if param_name in param_descriptions:
                param_schema["description"] = param_descriptions[param_name]
    
    # Step 5: Assemble final tool definition
    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": cleaned_description,
            "parameters": schema,
        },
    }


def functions_to_tools(functions: List[Callable]) -> List[Dict[str, Any]]:
    """
    Convert a list of Python functions to OpenAI-compatible tool definitions.
    
    :param functions: List of Python functions to convert
    :return: List of OpenAI-compatible tool definition dictionaries
    :raises ValueError: If any function cannot be analyzed or converted
    """
    return [function_to_tool(func) for func in functions]