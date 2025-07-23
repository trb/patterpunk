# Ai-generated open-hands

import inspect
import re
from typing import get_type_hints, List, Dict, Any, Callable

from pydantic import create_model

try:
    from docstring_parser import parse as parse_docstring

    HAS_DOCSTRING_PARSER = True
except ImportError:
    HAS_DOCSTRING_PARSER = False


def function_to_tool(func: Callable) -> Dict[str, Any]:
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

    if not fields:
        model = create_model(func.__name__ + "Model")
    else:
        model = create_model(func.__name__ + "Model", **fields)

    schema = model.model_json_schema()

    # Clean up schema for OpenAI format
    if "title" in schema:
        del schema["title"]
    schema["additionalProperties"] = False

    # Ensure required field is always present (even if empty)
    if "required" not in schema:
        schema["required"] = []

    description, param_descriptions = _parse_function_docs(func)

    if param_descriptions and "properties" in schema:
        for param_name, param_schema in schema["properties"].items():
            if param_name in param_descriptions:
                param_schema["description"] = param_descriptions[param_name]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": schema,
        },
    }


def functions_to_tools(functions: List[Callable]) -> List[Dict[str, Any]]:
    return [function_to_tool(func) for func in functions]


def _parse_function_docs(func: Callable) -> tuple[str, Dict[str, str]]:
    if not func.__doc__:
        return func.__name__, {}

    if HAS_DOCSTRING_PARSER:
        return _parse_with_docstring_parser(func)
    else:
        return _simple_parse_docstring(func)


def _parse_with_docstring_parser(func: Callable) -> tuple[str, Dict[str, str]]:
    try:
        parsed = parse_docstring(func.__doc__)

        description = ""
        if parsed.short_description:
            description = parsed.short_description
        if parsed.long_description:
            if description:
                description += "\n\n" + parsed.long_description
            else:
                description = parsed.long_description

        description = _clean_description(description or func.__name__)

        param_descriptions = {}
        if parsed.params:
            for param in parsed.params:
                if param.description:
                    clean_desc = re.sub(r"\s+", " ", param.description.strip())
                    param_descriptions[param.arg_name] = clean_desc

        return description, param_descriptions

    except Exception:
        return _simple_parse_docstring(func)


def _simple_parse_docstring(func: Callable) -> tuple[str, Dict[str, str]]:
    docstring = func.__doc__.strip()

    lines = docstring.split("\n")

    args_start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*Args?\s*:", line, re.IGNORECASE):
            args_start = i
            break

    if args_start is None:
        return _clean_description(docstring), {}

    description_lines = lines[:args_start]
    description = "\n".join(description_lines).strip()
    description = _clean_description(description or func.__name__)

    param_descriptions = {}
    current_param = None
    current_desc_lines = []

    for line in lines[args_start + 1 :]:
        if re.match(r"^\s*\w+\s*:", line):
            if current_param and current_desc_lines:
                desc = " ".join(current_desc_lines).strip()
                param_descriptions[current_param] = desc
            break

        param_match = re.match(r"^\s*(\w+)\s*:\s*(.+)", line)
        if param_match:
            if current_param and current_desc_lines:
                desc = " ".join(current_desc_lines).strip()
                param_descriptions[current_param] = desc

            current_param = param_match.group(1)
            current_desc_lines = [param_match.group(2)]
        elif current_param and line.strip():
            current_desc_lines.append(line.strip())

    if current_param and current_desc_lines:
        desc = " ".join(current_desc_lines).strip()
        param_descriptions[current_param] = desc

    return description, param_descriptions


def _clean_description(description: str) -> str:
    if not description:
        return ""

    sections_pattern = r"\n\s*(Args?|Arguments?|Parameters?|Param|Returns?|Return|Yields?|Yield|Raises?|Raise|Examples?|Example|Notes?|Note)\s*:.*$"
    cleaned = re.sub(
        sections_pattern,
        "",
        description,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )

    cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
    cleaned = re.sub(r"^\s+|\s+$", "", cleaned)

    return cleaned or description
