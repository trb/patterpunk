import re
from typing import Callable, Tuple, Dict


def parse_with_regex(func: Callable) -> Tuple[str, Dict[str, str]]:
    if not func.__doc__:
        return func.__name__, {}

    docstring = func.__doc__.strip()
    lines = docstring.split("\n")

    args_start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*Args?\s*:", line, re.IGNORECASE):
            args_start = i
            break

    if args_start is None:
        from ..cleaning import clean_description

        return clean_description(docstring), {}

    description_lines = lines[:args_start]
    description = "\n".join(description_lines).strip()

    param_descriptions = {}
    current_param = None
    current_desc_lines = []

    for line in lines[args_start + 1 :]:
        if re.match(
            r"^\s*(Returns?|Raises?|Yields?|Warns?|Warnings?|Note|Notes|Example|Examples|See Also|References?|Attributes?)\s*:",
            line,
            re.IGNORECASE,
        ):
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

    from ..cleaning import clean_description

    return clean_description(description) or func.__name__, param_descriptions
