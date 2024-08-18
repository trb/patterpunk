import copy
import json
from typing import Dict
from jinja2 import Template

from patterpunk.logger import logger


ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_FUNCTION_CALL = "function_call"


class BadParameterError(Exception):
    pass


class UnexpectedFunctionCallError(Exception):
    pass


class Message:
    def __init__(self, content: str, role: str = ROLE_USER):
        self.content = content
        self.role = role
        self._model = None
        self.is_function_call = False

    def format_content(self, parameters):
        variables = {}
        for parameter_name in parameters:
            value = parameters[parameter_name]
            if callable(value):
                value = value()

            value = value

            variables[parameter_name] = value

        template = Template(self.content)
        self.content = template.render(variables)

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

    def to_dict(self):
        return {"role": self.role, "content": self.content}

    @property
    def model(self):
        return self._model

    def __repr__(self, truncate=True):
        if self.content:
            content = (
                self.content
                if len(self.content) < 50 or truncate is False
                else f"{self.content[:50]}..."
            )
        else:
            content = "null"
        return f'{self.role.capitalize()}Message("{content}")'

    def copy(self):
        return copy.deepcopy(self)


class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__(content, ROLE_SYSTEM)


class UserMessage(Message):
    def __init__(self, content: str):
        super().__init__(content, ROLE_USER)


class AssistantMessage(Message):
    def __init__(self, content: str):
        super().__init__(content, ROLE_ASSISTANT)


class FunctionCallMessage(Message):
    def __init__(self, content, function_call):
        super().__init__(content, ROLE_FUNCTION_CALL)
        self._function_call = function_call
        self.available_functions: Dict[str, callable] = {}
        self.is_function_call = True

    @property
    def function_call(self):
        return self._function_call

    def set_available_functions(self, available_functions: Dict[str, callable]):
        self.available_functions = available_functions

    def to_dict(self):
        return {
            "role": "function",
            "content": self.function_call["arguments"],
            "name": self.function_call["name"],
        }

    def execute_function_call(self):
        name = self._function_call["name"]
        if name not in self.available_functions:
            raise UnexpectedFunctionCallError(
                f"Model indicated call to {name}, but no such function exists: {self.available_functions}"
            )

        return self.available_functions[name](
            **json.loads(self._function_call["arguments"])
        )

    def __repr__(self, truncate=True):
        if self.content:
            content = (
                self.content
                if len(self.content) < 50 or truncate is False
                else f"{self.content[:50]}..."
            )
        else:
            content = "null"
        return f'FunctionCallMessage("{content}", "{self._function_call}")'
