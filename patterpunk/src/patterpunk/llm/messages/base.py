"""
Core Message class with essential functionality only.

This module contains the base Message class that provides fundamental message
functionality while delegating complex operations to specialized modules.
"""

import copy
from typing import Union, List, Optional, Any

from patterpunk.config import GENERATE_STRUCTURED_OUTPUT_PROMPT
from patterpunk.lib.structured_output import get_model_schema, has_model_schema
from patterpunk.logger import logger
from ..cache import CacheChunk
from ..multimodal import MultimodalChunk
from ..types import ContentType
from .roles import ROLE_USER
from .exceptions import BadParameterError
from .templating import format_content
from .cache import get_content_as_string, has_cacheable_content, get_cache_chunks
from .structured_output import parse_structured_output


class Message:
    """
    Base message class providing core functionality for all message types.
    
    Handles content management, model assignment, templating, and basic
    dictionary conversion while delegating complex operations to specialized modules.
    """
    
    def __init__(self, content: ContentType, role: str = ROLE_USER):
        self.content = content
        self.role = role
        self._model = None
        self.structured_output = None
        self._parsed_output = None

    def format_content(self, parameters):
        variables = {}
        for parameter_name in parameters:
            value = parameters[parameter_name]
            if callable(value):
                value = value()
            variables[parameter_name] = value

        if isinstance(self.content, str):
            self.content = format_content(self.content, variables)
        elif isinstance(self.content, list):
            format_content(self.content, variables)

    def set_model(self, model):
        """
        When using the chat feature, you can switch the model for later messages. This can be helpful to e.g. try
        gpt3.5 to generate JSON and switch to gpt-4 if it fails, or so compare outputs, etc.

        :param model: Model
        :return: Message
        """
        new_message = self.copy()
        new_message._model = model
        return new_message

    def prepare(self, parameters):
        """
        Prepares the message for sending to LLM.

        Creates a copy of the message where all placeholders were replaced.
        :param parameters:
        :return: Message
        """
        new_message = copy.deepcopy(self)

        try:
            new_message.format_content(parameters)
        except KeyError as error:
            logger.warning(
                "Preparing message got unexpected parameter. Check that the function argument names match the placeholders in the message and vice versa",
                exc_info=error,
            )
            raise BadParameterError(error)
        return new_message

    def get_content_as_string(self) -> str:
        return get_content_as_string(self.content)
    
    def has_cacheable_content(self) -> bool:
        return has_cacheable_content(self.content)
    
    def get_cache_chunks(self) -> List[Union[CacheChunk, MultimodalChunk]]:
        return get_cache_chunks(self.content)

    @property
    def parsed_output(self):
        return parse_structured_output(
            self.content, 
            self.structured_output, 
            self.role, 
            self._parsed_output
        )

    def to_dict(self, prompt_for_structured_output: bool = False):
        content = self.get_content_as_string()
        if (
            prompt_for_structured_output
            and self.structured_output
            and has_model_schema(self.structured_output)
        ):
            content = f"{content}\n{GENERATE_STRUCTURED_OUTPUT_PROMPT}{get_model_schema(self.structured_output)}"

        return {"role": self.role, "content": content}

    @property
    def model(self):
        return self._model

    def __repr__(self, truncate=True):
        content_str = self.get_content_as_string()
        if content_str:
            content = (
                content_str
                if len(content_str) < 50 or truncate is False
                else f"{content_str[:50]}..."
            )
        else:
            content = "null"
        return f'{self.role.capitalize()}Message("{content}")'

    def copy(self):
        return copy.deepcopy(self)