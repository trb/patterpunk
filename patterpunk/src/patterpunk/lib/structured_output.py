class ModelSchemaNotAvailable(Exception):
    pass


def has_model_schema(model: object) -> bool:
    if hasattr(model, "model_json_schema") and callable(
        getattr(model, "model_json_schema")
    ):
        return True

    if hasattr(model, "schema") and callable(getattr(model, "schema")):
        return True

    if hasattr(model, "model_schema") and callable(getattr(model, "model_schema")):
        return True

    return False


def get_model_schema(model: object):
    if hasattr(model, "model_json_schema") and callable(
        getattr(model, "model_json_schema")
    ):
        return model.model_json_schema()

    if hasattr(model, "schema") and callable(getattr(model, "schema")):
        return model.schema()

    if hasattr(model, "model_schema") and callable(getattr(model, "model_schema")):
        return model.model_schema()

    raise ModelSchemaNotAvailable(
        "The provided model does not support schema generation"
    )
