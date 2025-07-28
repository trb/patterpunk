import copy
from jinja2 import Template
from typing import List, Dict, Union

from patterpunk.config import GENERATE_STRUCTURED_OUTPUT_PROMPT
from patterpunk.lib.structured_output import get_model_schema, has_model_schema
from patterpunk.lib.extract_json import extract_json
from patterpunk.logger import logger
from patterpunk.llm.types import ToolCallList, CacheChunk


ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL_CALL = "tool_call"


class BadParameterError(Exception):
    pass


class UnexpectedFunctionCallError(Exception):
    pass


class StructuredOutputNotPydanticLikeError(Exception):
    pass


class StructuredOutputFailedToParseError(Exception):
    pass


class Message:
    def __init__(self, content: Union[str, List[CacheChunk]], role: str = ROLE_USER):
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

            value = value

            variables[parameter_name] = value

        if isinstance(self.content, str):
            template = Template(self.content)
            self.content = template.render(variables)
        elif isinstance(self.content, list):
            # Handle template rendering for each chunk
            for chunk in self.content:
                template = Template(chunk.content)
                chunk.content = template.render(variables)

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
        """Helper method to get content as string for backward compatibility."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            return "".join(chunk.content for chunk in self.content)
        else:
            return str(self.content)
    
    def has_cacheable_content(self) -> bool:
        """Check if message contains any cacheable chunks."""
        if isinstance(self.content, list):
            return any(chunk.cacheable for chunk in self.content)
        return False
    
    def get_cache_chunks(self) -> List[CacheChunk]:
        """Get cache chunks, converting string content to non-cacheable chunk if needed."""
        if isinstance(self.content, str):
            return [CacheChunk(content=self.content, cacheable=False)]
        elif isinstance(self.content, list):
            return self.content
        else:
            return [CacheChunk(content=str(self.content), cacheable=False)]

    @property
    def parsed_output(self):
        if self._parsed_output is not None:
            return self._parsed_output

        if not self.structured_output:
            return None

        if not getattr(self.structured_output, "parse_raw", None) and not getattr(
            self.structured_output, "model_validate_json", None
        ):
            raise StructuredOutputNotPydanticLikeError(
                f"[MESSAGE][{self.role}] The provided structured_output is not a pydantic model (missing parse_raw or model_validate_json)"
            )

        json_messages = extract_json(self.get_content_as_string())
        for json_message in json_messages:
            try:
                if getattr(self.structured_output, "model_validate_json", None):
                    obj = self.structured_output.model_validate_json(json_message)
                else:
                    obj = self.structured_output.parse_raw(json_message)

                self._parsed_output = obj

                return obj
            except Exception as error:
                logger.debug(
                    f"[MESSAGE][{self.role}][STRUCTURED_OUTPUT] Failed to parse response {json_message}: {error}",
                    exc_info=error,
                )

        raise StructuredOutputFailedToParseError(
            f"[MESSAGE][{self.role}] Structured output could not be parsed, message: \n{self.get_content_as_string()}"
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


class SystemMessage(Message):
    def __init__(self, content: Union[str, List[CacheChunk]]):
        super().__init__(content, ROLE_SYSTEM)


class UserMessage(Message):
    def __init__(self, content: Union[str, List[CacheChunk]], structured_output=None, allow_tool_calls=True):
        super().__init__(content, ROLE_USER)
        self.structured_output = structured_output
        self.allow_tool_calls = allow_tool_calls


class AssistantMessage(Message):
    def __init__(self, content: str, structured_output=None, parsed_output=None):
        super().__init__(content, ROLE_ASSISTANT)
        self.structured_output = structured_output
        self._parsed_output = parsed_output


class ToolCallMessage(Message):
    def __init__(self, tool_calls: ToolCallList):
        """
        Represents a tool call message from the LLM.

        :param tool_calls: List of tool calls, each containing id, function name, and arguments
        """
        super().__init__("", ROLE_TOOL_CALL)
        self.tool_calls = tool_calls

    def to_dict(self, prompt_for_structured_output: bool = False):
        return {"role": self.role, "tool_calls": self.tool_calls}

    def __repr__(self, truncate=True):
        tool_names = [
            call.get("function", {}).get("name", "unknown") for call in self.tool_calls
        ]
        return f'ToolCallMessage({", ".join(tool_names)})'
