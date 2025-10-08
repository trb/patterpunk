from jinja2 import Template
from typing import Union, List, Dict, Any

from ..chunks import CacheChunk, TextChunk


def format_content(
    content: Union[str, List[Union[TextChunk, CacheChunk]]], parameters: Dict[str, Any]
) -> None:
    variables = {}
    for parameter_name in parameters:
        value = parameters[parameter_name]
        if callable(value):
            value = value()
        variables[parameter_name] = value

    if isinstance(content, str):
        template = Template(content)
        return template.render(variables)
    elif isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, (TextChunk, CacheChunk)):
                template = Template(chunk.content)
                chunk.content = template.render(variables)
        return content

    return content
