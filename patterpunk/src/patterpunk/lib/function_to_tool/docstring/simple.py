"""
Regex-based docstring parsing fallback.

This module provides a simple but robust regex-based parser that handles
basic docstring formats when the advanced parser isn't available.
"""

import re
from typing import Callable, Tuple, Dict


def parse_with_regex(func: Callable) -> Tuple[str, Dict[str, str]]:
    """
    Parse function docstring using simple regex patterns.
    
    Handles basic docstring formats with Args: sections and parameter descriptions.
    
    :param func: Function to parse docstring from
    :return: Tuple of (description, parameter_descriptions)
    """
    if not func.__doc__:
        return func.__name__, {}
        
    docstring = func.__doc__.strip()
    lines = docstring.split("\n")

    # Find the Args section
    args_start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*Args?\s*:", line, re.IGNORECASE):
            args_start = i
            break

    if args_start is None:
        # No Args section found, return the entire docstring as description
        from .cleaning import clean_description
        return clean_description(docstring), {}

    # Extract description (everything before Args section)
    description_lines = lines[:args_start]
    description = "\n".join(description_lines).strip()
    
    # Extract parameter descriptions
    param_descriptions = {}
    current_param = None
    current_desc_lines = []

    for line in lines[args_start + 1:]:
        # Stop if we hit another section (like Returns:, Raises:, etc.)
        if re.match(r"^\s*\w+\s*:", line):
            # Save the current parameter if we have one
            if current_param and current_desc_lines:
                desc = " ".join(current_desc_lines).strip()
                param_descriptions[current_param] = desc
            break

        # Check for parameter definition (param_name: description)
        param_match = re.match(r"^\s*(\w+)\s*:\s*(.+)", line)
        if param_match:
            # Save previous parameter if we have one
            if current_param and current_desc_lines:
                desc = " ".join(current_desc_lines).strip()
                param_descriptions[current_param] = desc

            current_param = param_match.group(1)
            current_desc_lines = [param_match.group(2)]
        elif current_param and line.strip():
            # Continue multi-line description for current parameter
            current_desc_lines.append(line.strip())

    # Don't forget the last parameter
    if current_param and current_desc_lines:
        desc = " ".join(current_desc_lines).strip()
        param_descriptions[current_param] = desc

    from .cleaning import clean_description
    return clean_description(description) or func.__name__, param_descriptions