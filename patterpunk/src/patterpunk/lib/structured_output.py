class ModelSchemaNotAvailable(Exception):
    pass


def has_model_schema(model: object) -> bool:
    # Check for Pydantic v2 model_json_schema method
    if hasattr(model, "model_json_schema") and callable(getattr(model, "model_json_schema")):
        return True

    # Check for Pydantic v1 schema method
    if hasattr(model, "schema") and callable(getattr(model, "schema")):
        return True

    # Check for Pydantic v1 model_schema method
    if hasattr(model, "model_schema") and callable(getattr(model, "model_schema")):
        return True

    return False

def get_model_schema(model: object):
    # Check for Pydantic v2 model_json_schema method
    if hasattr(model, "model_json_schema") and callable(getattr(model, "model_json_schema")):
        return model.model_json_schema()

    # Check for Pydantic v1 schema method
    if hasattr(model, "schema") and callable(getattr(model, "schema")):
        return model.schema()

    # Check for Pydantic v1 model_schema method
    if hasattr(model, "model_schema") and callable(getattr(model, "model_schema")):
        return model.model_schema()

    # No schema generation method found
    raise ModelSchemaNotAvailable("The provided model does not support schema generation")
