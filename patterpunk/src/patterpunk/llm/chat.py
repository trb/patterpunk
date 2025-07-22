import copy
import re
from typing import Dict, List, Optional, _GenericAlias

from patterpunk.lib.extract_json import extract_json
from patterpunk.llm.defaults import default_model
from patterpunk.llm.messages import (
    AssistantMessage,
    ToolCallMessage,
    Message,
    StructuredOutputFailedToParseError,
    UserMessage,
)
from patterpunk.llm.models.openai import Model
from patterpunk.logger import logger


class StructuredOutputParsingError(Exception):
    pass


class Chat:
    def __init__(
        self,
        messages: Optional[List[Message]] = None,
        model: Optional[Model] = None,
        tools: Optional[list] = None,
    ):
        if messages is None:
            messages = []
        self.messages = messages

        self.model = default_model() if model is None else model
        self.tools = tools

    def add_message(self, message: Message):
        new_chat = self.copy()
        new_chat.messages.append(message)
        return new_chat
    
    def with_tools(self, tools: list):
        """
        Set tools available for this chat.
        
        :param tools: List of tool definitions in OpenAI format
        :return: New Chat instance with tools set
        """
        new_chat = self.copy()
        new_chat.tools = tools
        return new_chat

    def complete(self):
        message = self.latest_message
        model = message.model if message.model else self.model
        
        # Only pass tools if the latest message allows tool calls
        tools_to_use = None
        if self.tools and getattr(message, "allow_tool_calls", True):
            tools_to_use = self.tools
            
        response_message = model.generate_assistant_message(
            self.messages,
            tools_to_use,
            structured_output=getattr(message, "structured_output", None),
        )

        return self.add_message(response_message)

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
    def is_latest_message_tool_call(self):
        return isinstance(self.latest_message, ToolCallMessage)

    def copy(self):
        return copy.deepcopy(self)
