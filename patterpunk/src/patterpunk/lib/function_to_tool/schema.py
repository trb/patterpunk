from typing import Dict, Tuple, Any
from pydantic import create_model


def create_pydantic_model_from_fields(
    function_name: str, fields: Dict[str, Tuple]
) -> Any:
    model_name = function_name + "Model"

    if not fields:
        model = create_model(model_name)
    else:
        model = create_model(model_name, **fields)

    return model


def generate_openai_compatible_schema(model: Any) -> Dict[str, Any]:
    schema = model.model_json_schema()

    if "title" in schema:
        del schema["title"]
    schema["additionalProperties"] = False

    if "required" not in schema:
        schema["required"] = []

    return schema
