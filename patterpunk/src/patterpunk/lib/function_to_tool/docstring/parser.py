from typing import Callable, Tuple, Dict

from .advanced import parse_with_docstring_parser, is_advanced_parser_available
from .simple import parse_with_regex


def parse_function_docs(func: Callable) -> Tuple[str, Dict[str, str]]:
    if not func.__doc__:
        return func.__name__, {}

    if is_advanced_parser_available():
        try:
            return parse_with_docstring_parser(func)
        except (ImportError, ValueError):
            pass

    return parse_with_regex(func)
