from typing import Callable, Tuple, Dict

try:
    from docstring_parser import parse as parse_docstring

    HAS_DOCSTRING_PARSER = True
except ImportError:
    HAS_DOCSTRING_PARSER = False


def parse_with_docstring_parser(func: Callable) -> Tuple[str, Dict[str, str]]:
    if not HAS_DOCSTRING_PARSER:
        raise ImportError("docstring_parser library not available")

    if not func.__doc__:
        return func.__name__, {}

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

        param_descriptions = {}
        if parsed.params:
            for param in parsed.params:
                if param.description:
                    import re

                    clean_desc = re.sub(r"\s+", " ", param.description.strip())
                    param_descriptions[param.arg_name] = clean_desc

        return description or func.__name__, param_descriptions

    except Exception as e:
        raise ValueError(f"Advanced docstring parsing failed: {e}")


def is_advanced_parser_available() -> bool:
    return HAS_DOCSTRING_PARSER
