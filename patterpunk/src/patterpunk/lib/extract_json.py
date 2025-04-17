from typing import List


def extract_json(json_string: str) -> List[str]:
    json_substrings = []
    stack = []
    inside_string = False
    start_index = None
    bracket_type = None

    for i, char in enumerate(json_string):
        if (
            char in ["{", "["]
            and not inside_string
            and (not bracket_type or bracket_type == char)
        ):
            if not stack:
                start_index = i
                bracket_type = char
            stack.append(char)
        elif (
            char in ["}", "]"] and not inside_string and stack and bracket_type == "{"
            if char == "}"
            else bracket_type == "["
        ):
            stack.pop()
            if not stack:
                json_substrings.append(json_string[start_index : i + 1])
                bracket_type = None
        elif char == '"':
            if not inside_string or (inside_string and json_string[i - 1] != "\\"):
                inside_string = not inside_string

    return json_substrings
