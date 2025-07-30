"""
Pydantic model creation and JSON schema generation.

This module handles the complex logic of creating dynamic Pydantic models
from function signatures and generating OpenAI-compatible JSON schemas.
"""

from typing import Dict, Tuple, Any
from pydantic import create_model


def create_pydantic_model_from_fields(function_name: str, fields: Dict[str, Tuple]) -> Any:
    """
    Create a dynamic Pydantic model from function parameter fields.
    
    :param function_name: Name of the function (used for model name)
    :param fields: Dictionary mapping parameter names to (type, default) tuples
    :return: Dynamically created Pydantic model class
    """
    model_name = function_name + "Model"
    
    if not fields:
        model = create_model(model_name)
    else:
        model = create_model(model_name, **fields)
    
    return model


def generate_openai_compatible_schema(model: Any) -> Dict[str, Any]:
    """
    Generate OpenAI-compatible JSON schema from Pydantic model.
    
    :param model: Pydantic model class
    :return: JSON schema dictionary compatible with OpenAI function calling
    """
    schema = model.model_json_schema()

    if "title" in schema:
        del schema["title"]
    schema["additionalProperties"] = False

    if "required" not in schema:
        schema["required"] = []
    
    return schema