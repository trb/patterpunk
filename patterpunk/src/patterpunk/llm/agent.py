from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Generic, TypeVar, get_args

import jinja2
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.base import Model

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class Agent(Generic[InputType, OutputType], ABC):
    @property
    @abstractmethod
    def model(self) -> Model:
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    @abstractmethod
    def _user_prompt_template(self) -> str:
        pass

    def execute(self, input_data: InputType) -> OutputType:
        chat = self.prepare_chat()
        user_message = self.prepare_user_message(input_data)
        chat = chat.add_message(user_message)
        chat = chat.complete()
        return self.process_response(chat)

    def prepare_chat(self) -> Chat:
        chat = Chat(
            model=self.model,
        )
        return chat.add_message(SystemMessage(self.system_prompt))

    def prepare_user_message(self, input_data: InputType) -> UserMessage:
        user_prompt = self._render_user_prompt(input_data)
        output_type = self._get_output_type()
        structured_output = None if output_type is str else output_type
        return UserMessage(
            user_prompt,
            structured_output=structured_output,
        )

    def process_response(self, chat: Chat) -> OutputType:
        output_type = self._get_output_type()
        if output_type is str:
            return chat.latest_message.content
        return chat.parsed_output

    def _get_output_type(self):
        return get_args(self.__class__.__orig_bases__[0])[1]

    def _render_user_prompt(self, input_data: InputType) -> str:
        template = jinja2.Template(self._user_prompt_template)

        if isinstance(input_data, str):
            return template.render(text=input_data)
        else:
            return template.render(asdict(input_data))
