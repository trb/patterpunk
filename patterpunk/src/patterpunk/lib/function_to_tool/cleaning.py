"""
Description and schema cleaning utilities.

This module contains specialized text processing for cleaning descriptions
and removing unwanted docstring sections.
"""

import re


def clean_description(description: str) -> str:
    """
    Clean description text by removing docstring sections and normalizing whitespace.
    
    Removes sections like Args, Returns, Raises, Examples, etc. and normalizes
    whitespace while preserving the core description content.
    
    :param description: Raw description text to clean
    :return: Cleaned description text
    """
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