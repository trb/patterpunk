import enum
from abc import ABC
from typing import List, Literal, Optional, Union

from patterpunk.config import DEFAULT_TEMPERATURE, openai, OPENAI_MAX_RETRIES
from patterpunk.lib.structured_output import has_model_schema
from patterpunk.llm.messages import AssistantMessage, Message, ToolCallMessage
from patterpunk.llm.models.base import Model
from patterpunk.llm.types import ToolDefinition
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


class OpenAiReasoningEffort(enum.Enum):
    LOW = enum.auto()
    MEDIUM = enum.auto()
    HIGH = enum.auto()


class OpenAiModel(Model, ABC):
    def __init__(
        self,
        model="",
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        logit_bias=None,
        reasoning_effort: Optional[
            Union[
                OpenAiReasoningEffort,
                Literal["low", "medium", "high"],
            ]
        ] = OpenAiReasoningEffort.LOW,
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

        if not isinstance(reasoning_effort, OpenAiReasoningEffort):
            try:
                reasoning_effort = OpenAiReasoningEffort[reasoning_effort.upper()]
            except KeyError:
                message = f"Reasoning effort must be a value of OpenAiReasoningEffort or 'low', 'medium' or 'high', parameter given: {reasoning_effort}"
                logger.warning(message)
                raise OpenAiWrongParameterError(message, "reasoning_effort")

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logit_bias = {} if logit_bias is None else logit_bias
        self.completion = None
        self.reasoning_effort = reasoning_effort

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
    ) -> Union[AssistantMessage, ToolCallMessage]:
        logger.info("Request to OpenAi made")
        logger_llm.debug(
            "\n---\n".join(
                [f"{message.__repr__(truncate=False)}" for message in messages]
            )
        )

        openai_parameters = {}
        openai_call = openai.chat.completions.create
        set_parsed_output = False
        prompt_for_structured_output = False

        if structured_output and has_model_schema(structured_output):
            # Fall back to JSON mode for models that don't support structured output
            if (
                self.model.startswith("o")
                or self.model.startswith("gpt-4o")
                or self.model.startswith("gpt-4.")
            ):
                openai_call = openai.beta.chat.completions.parse
                set_parsed_output = True
                openai_parameters["response_format"] = structured_output
            else:
                openai_parameters["response_format"] = {
                    "type": "json_object",
                }

                prompt_for_structured_output = True

        openai_parameters["model"] = self.model
        openai_parameters["messages"] = [
            message.to_dict(prompt_for_structured_output=prompt_for_structured_output)
            for message in messages
        ]
        
        # Add tools if provided
        if tools:
            openai_parameters["tools"] = tools

        if self.model.startswith("o"):
            openai_parameters["reasoning_effort"] = self.reasoning_effort.name.lower()
        else:
            openai_parameters = {
                **openai_parameters,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "logit_bias": self.logit_bias,
            }

        # Log only the parameters that are actually sent to the model
        log_params = {k: v for k, v in openai_parameters.items() if k != "messages"}
        param_strings = []
        for key, value in log_params.items():
            param_strings.append(f"{key}: {value}")
        logger_llm.info(f"OpenAi Model params: {', '.join(param_strings)}")

        retry_count = 0
        done = False
        response = False

        while not done and retry_count < OPENAI_MAX_RETRIES:
            try:
                response = openai_call(**openai_parameters)
                logger.info("OpenAi response received")
                done = True
            except APIError as error:
                logger.info("Retrying OpenAi request due to APIError", exc_info=error)
                retry_count += 1

        if not done or not response:
            raise OpenAiApiError("OpenAi api is returning too many api errors")

        response_message = response.choices[0]
        logger_llm.info(f"[Assistant]\n{response_message}")

        # Check if the response contains tool calls
        if response_message.message.tool_calls:
            # Convert OpenAI tool calls to our format
            tool_calls = []
            for tool_call in response_message.message.tool_calls:
                tool_calls.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
            return ToolCallMessage(tool_calls)
        
        return AssistantMessage(
            response_message.message.content,
            structured_output=structured_output,
            parsed_output=(
                response_message.message.parsed if set_parsed_output else None
            ),
        )

    @staticmethod
    def get_name():
        return "OpenAI"

    @staticmethod
    def get_available_models() -> List[str]:
        return [model.id for model in openai.models.list().data]
