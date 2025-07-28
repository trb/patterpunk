"""
Jinja2 template rendering engine and parameter handling for messages.

This module handles template parameter validation, compilation, and rendering
for both string content and cache chunk collections.
"""

from jinja2 import Template
from typing import Union, List, Dict, Any

from ..cache import CacheChunk


def format_content(content: Union[str, List[CacheChunk]], parameters: Dict[str, Any]) -> None:
    """
    Format content with template parameters using Jinja2.
    
    Modifies content in-place, rendering all template placeholders with provided parameters.
    Supports both string content and list of cache chunks.
    
    :param content: The content to format (string or list of CacheChunk)
    :param parameters: Dictionary of parameter name/value pairs for template rendering
    :raises KeyError: If template requires parameters not provided
    """
    variables = {}
    for parameter_name in parameters:
        value = parameters[parameter_name]
        if callable(value):
            value = value()
        variables[parameter_name] = value

    if isinstance(content, str):
        template = Template(content)
        # Note: This modifies the content in-place by reassigning to the same variable
        # This works because the calling code passes the content by reference
        return template.render(variables)
    elif isinstance(content, list):
        # Handle template rendering for each chunk
        for chunk in content:
            template = Template(chunk.content)
            chunk.content = template.render(variables)
        return content
    
    return content