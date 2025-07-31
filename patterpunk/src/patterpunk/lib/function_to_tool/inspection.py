import inspect
from typing import get_type_hints, Callable, Dict, Any, Tuple


def analyze_function_signature(
    func: Callable,
) -> Tuple[inspect.Signature, Dict[str, Any], Dict[str, Tuple]]:
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError) as e:
        func_name = getattr(func, "__name__", str(func))
        raise ValueError(f"Cannot inspect function signature for {func_name}: {e}")

    type_hints = get_type_hints(func)

    fields = {}
    for name, param in signature.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        param_type = type_hints.get(name, str)

        if param.default != inspect.Parameter.empty:
            fields[name] = (param_type, param.default)
        else:
            fields[name] = (param_type, ...)

    return signature, type_hints, fields


def get_function_name(func: Callable) -> str:
    return getattr(func, "__name__", str(func))
