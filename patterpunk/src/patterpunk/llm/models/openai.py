from abc import ABC
from typing import List

from patterpunk.config import DEFAULT_TEMPERATURE, OPENAI_MAX_RETRIES, openai
from patterpunk.llm.messages import AssistantMessage, FunctionCallMessage, Message
from patterpunk.llm.models.base import Model
from patterpunk.logger import logger, logger_llm

if openai:
    from openai import APIError


class OpenAiWrongParameterError(Exception):
    def __init__(self, message, parameter: str = ""):
        super().__init__(message)
        self.parameter = parameter


class OpenAiMissingConfigurationError(Exception):
    pass


class OpenAiApiError(Exception):
    pass


class OpenAiModel(Model, ABC):
    def __init__(
        self,
        model="",
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        logit_bias=None,
    ):
        if not openai:
            raise OpenAiMissingConfigurationError(
                "OpenAi was not initialized correctly, did you set the api key?"
            )

        if temperature is None:
            temperature = DEFAULT_TEMPERATURE
        if top_p is None:
            top_p = 1.0
        if frequency_penalty is None:
            frequency_penalty = 0.0
        if presence_penalty is None:
            presence_penalty = 0.0

        if not 0.0 <= temperature <= 2.0:
            message = (
                f"temperature needs to be between 0 and 2, {temperature} was given"
            )
            logger.warning(message)
            raise OpenAiWrongParameterError(message, "temperature")

        if not 0.0 <= top_p <= 1.0:
            message = f"top_p needs to be between 0 and 1, {top_p} was given"
            logger.warning(message)
            raise OpenAiWrongParameterError(message, "top_p")

        if not -2.0 <= frequency_penalty <= 2.0:
            message = f"frequency_penalty needs to be between 0 and 2, {frequency_penalty} was given"
            logger.warning(message)
            raise OpenAiWrongParameterError(message, "frequency_penalty")

        if not -2.0 <= presence_penalty <= 2.0:
            message = f"presence_penalty needs to be between 0 and 2, {presence_penalty} was given"
            logger.warning(message)
            raise OpenAiWrongParameterError(message, "presence_penalty")

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logit_bias = {} if logit_bias is None else logit_bias

        self.completion = None

    def generate_assistant_message(
        self, messages: List[Message], functions: list | None = None
    ) -> AssistantMessage | FunctionCallMessage:
        logger.info("Request to OpenAi made")
        logger_llm.debug(
            "\n---\n".join(
                [f"{message.__repr__(truncate=False)}" for message in messages]
            )
        )
        logger_llm.info(
            f"Model params: {self.model}, temp: {self.temperature}, top_p: {self.top_p}, frequency_penalty: {self.frequency_penalty}, presence_penalty: {self.presence_penalty}, functions: {functions}"
        )

        openai_parameters = {
            "model": self.model,
            # don't send function calls back, they're wasted tokens
            "messages": [
                message.to_dict()
                for message in messages
                if not message.is_function_call
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "logit_bias": self.logit_bias,
        }

        if functions:
            openai_parameters["functions"] = functions

        retry_count = 1
        done = False
        response = False

        while not done and retry_count < OPENAI_MAX_RETRIES:
            try:
                response = openai.chat.completions.create(**openai_parameters)
                logger.info("OpenAi response received")
                done = True
            except APIError as error:
                logger.info("Retrying OpenAI request due to APIError", exc_info=error)
                retry_count += 1

        if not done or not response:
            raise OpenAiApiError("OpenAi api is returning too many api errors")

        response_message = response.choices[0]
        logger_llm.info(f"[Assistant]\n{response_message}")

        if response_message.finish_reason == "function_call":
            return FunctionCallMessage(
                response_message.message.content,
                response_message.message.function_call,
            )
        else:
            return AssistantMessage(response_message.message.content)

    @staticmethod
    def get_name():
        return "OpenAI"

    @staticmethod
    def get_available_models() -> List[str]:
        return [model.id for model in openai.models.list().data]
