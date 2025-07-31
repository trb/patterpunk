import re


def clean_description(description: str) -> str:
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
