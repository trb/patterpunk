from typing import Callable, Dict, Any, List

from .inspection import analyze_function_signature, get_function_name
from .schema import create_pydantic_model_from_fields, generate_openai_compatible_schema
from .docstring.parser import parse_function_docs
from .cleaning import clean_description


def function_to_tool(func: Callable) -> Dict[str, Any]:
    signature, type_hints, fields = analyze_function_signature(func)
    function_name = get_function_name(func)

    model = create_pydantic_model_from_fields(function_name, fields)
    schema = generate_openai_compatible_schema(model)

    description, param_descriptions = parse_function_docs(func)
    cleaned_description = clean_description(description)

    if param_descriptions and "properties" in schema:
        for param_name, param_schema in schema["properties"].items():
            if param_name in param_descriptions:
                param_schema["description"] = param_descriptions[param_name]
    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": cleaned_description,
            "parameters": schema,
        },
    }


def functions_to_tools(functions: List[Callable]) -> List[Dict[str, Any]]:
    return [function_to_tool(func) for func in functions]
