import copy
import dataclasses
import inspect
import re
from types import GenericAlias, UnionType
from typing import Dict, List, Optional, _GenericAlias

from patterpunk.lib.extract_json import extract_json
from patterpunk.llm.defaults import default_model
from patterpunk.llm.messages import (
    AssistantMessage,
    FunctionCallMessage,
    Message,
    StructuredOutputFailedToParseError,
    UserMessage,
)
from patterpunk.llm.models.openai import Model
from patterpunk.logger import logger


class UnsupportedParameterTypeError(Exception):
    pass


class NotAFunctionCallError(Exception):
    pass


class NotAFunctionError(Exception):
    pass


class FunctionNotFoundError(Exception):
    pass


class StructuredOutputParsingError(Exception):
    pass


def type_to_json_schema(type_, name=None):
    prop = {}
    if name:
        prop["title"] = name
    if type_ is int:
        prop["type"] = "number"
    elif type_ is bool:
        prop["type"] = "bool"
    elif type_ is float:
        prop["type"] = "float"
    elif type_ is str:
        prop["type"] = "string"
    elif isinstance(type_, UnionType):
        prop["type"] = "oneof"
        prop["properties"] = [
            type_to_json_schema(union_prop) for union_prop in type_.__args__
        ]
    elif hasattr(type_, "model_json_schema"):
        prop = type_.model_json_schema()
    elif (
        isinstance(type_, _GenericAlias)
        or isinstance(type_, GenericAlias)
        and hasattr(type_, "__origin__")
    ):
        if type_.__origin__ is list:
            prop["type"] = "array"
            prop["items"] = type_to_json_schema(type_.__args__[0])
        elif type_.__origin__ is dict:
            prop["type"] = "object"
            if len(type_.__args__) == 1:
                prop["additionalProperties"] = type_to_json_schema(type_.__args__[0])
            else:
                prop["additionalProperties"] = type_to_json_schema(type_.__args__[1])
        else:
            message = f"Unsupported parameter type to decorated function found on parameter {name}, type: {type(type_)}"
            logger.critical(message)
            raise UnsupportedParameterTypeError(message)
    elif dataclasses.is_dataclass(type_):
        prop["type"] = "object"
        if name:
            prop["name"] = name
        prop["additionalProperties"] = []
        for field in dataclasses.fields(type_):
            prop["additionalProperties"].append(
                type_to_json_schema(field.type, field.name)
            )
    elif isinstance(type_, object):
        message = f"Classes as parameters need to inherit from pydantics BaseModel or implement a static method named `model_json_schema`, parameter: {name}"
        logger.critical(message)
        raise UnsupportedParameterTypeError(message)
    else:
        message = f"Unknown parameter type encountered for {name}, type {type(type_)}"
        logger.critical(message)
        raise UnsupportedParameterTypeError(message)

    return prop


def serialize_functions(functions: list):
    new_functions = []
    for function in functions:
        signature = inspect.signature(function)

        doc_string = []
        doc_params = {}
        if function.__doc__:
            for line in function.__doc__.split("\n"):
                line = line.strip()
                if line.startswith(":"):
                    if line.startswith(":param"):
                        match = re.match(r":param (\w+): (.*)", line)

                        if match:
                            name, doc = match.groups()
                            doc_params[name] = doc
                else:
                    doc_string.append(line)

        parameters = {"type": "object", "properties": {}}
        for parameter_name in signature.parameters:
            parameter = type_to_json_schema(
                signature.parameters[parameter_name].annotation, parameter_name
            )
            parameter["description"] = (
                doc_params[parameter_name] if parameter_name in doc_params else ""
            )
            parameters["properties"][parameter_name] = parameter

        function_signature = {
            "name": function.__name__,
            "description": "\n".join([line for line in doc_string if line]),
            "parameters": parameters,
            "required": list(signature.parameters.keys()),
        }

        new_functions.append(function_signature)

    return new_functions


class Chat:
    def __init__(
        self,
        messages: Optional[List[Message]] = None,
        model: Optional[Model] = None,
        functions: Optional[list] = None,
    ):
        if messages is None:
            messages = []
        self.messages = messages

        self.model = default_model() if model is None else model
        self.functions_map: Dict[str, callable] = {}
        self.functions = None
        if functions:
            for function in functions:
                if not callable(function):
                    raise NotAFunctionError(
                        f"Chat received `functions`, but one of the items is not callable. Maybe a variable has the same name as the function? Problematic functions item: {function}, Functions: {functions}"
                    )
                self.functions_map[function.__name__] = function

            self.functions = serialize_functions(functions) if functions else None

    def add_message(self, message: Message):
        new_chat = self.copy()
        new_chat.messages.append(message)
        return new_chat

    def complete(self):
        message = self.latest_message
        model = message.model if message.model else self.model
        response_message = model.generate_assistant_message(
            self.messages,
            self.functions,
            structured_output=getattr(message, "structured_output", None),
        )
        if hasattr(response_message, "set_available_functions"):
            response_message.set_available_functions(self.functions_map)

        return self.add_message(response_message)

    def set_functions(self, functions: list):
        new_chat = self.copy()
        new_chat.functions = serialize_functions(functions)
        return new_chat

    def extract_json(self) -> Optional[List[str]]:
        """Extracts any json in any non-user message"""
        chat_text = "\n\n".join(
            [
                f"{message.role}:\n{message.content}"
                for message in self.messages
                # we don't want to extract JSON the user send
                if isinstance(message, AssistantMessage)
            ]
        )

        if re.search(r"\{.*}", chat_text, re.IGNORECASE | re.DOTALL) or re.search(
            r"\[.*]", chat_text, re.IGNORECASE | re.DOTALL
        ):
            jsons = extract_json(chat_text)
            if jsons:
                return jsons

        return None

    @property
    def parsed_output(self):
        if not getattr(self.latest_message, "structured_output", None):
            return None

        retry = 0
        max_reties = 2

        chat = self

        while retry < max_reties:
            try:
                obj = chat.latest_message.parsed_output
                return obj
            except StructuredOutputFailedToParseError as error:
                logger.debug(
                    "[CHAT] Failed to parse structured_output from latest message",
                    exc_info=error,
                )
                chat = chat.add_message(
                    UserMessage(
                        "You did not generate valid JSON! YOUR RESPONSE HAS TO BE A VALID JSON OBJECT THAT CONFORMS TO THE JSON SCHEMA!",
                        structured_output=chat.latest_message.structured_output,
                    )
                ).complete()
                retry += 1

        raise StructuredOutputParsingError(
            f"[CHAT] Failed to parse structured_output from latest message, latest message:\n{self.latest_message.content}"
        )

    @property
    def latest_message(self):
        return self.messages[-1]

    @property
    def is_latest_message_function_call(self):
        return isinstance(self.latest_message, FunctionCallMessage)

    def copy(self):
        return copy.deepcopy(self)
