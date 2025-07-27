import enum
from abc import ABC
from typing import List, Literal, Optional, Union

from patterpunk.config import DEFAULT_TEMPERATURE, openai, OPENAI_MAX_RETRIES
from patterpunk.lib.structured_output import has_model_schema, get_model_schema
from patterpunk.lib.extract_json import extract_json
from patterpunk.llm.messages import AssistantMessage, Message, ToolCallMessage
from patterpunk.llm.models.base import Model
from patterpunk.llm.thinking import ThinkingConfig
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
        thinking_config: Optional[ThinkingConfig] = None,
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

        reasoning_effort = OpenAiReasoningEffort.LOW
        if thinking_config is not None:
            if thinking_config.effort is not None:
                reasoning_effort = OpenAiReasoningEffort[thinking_config.effort.upper()]
            else:
                if thinking_config.token_budget == 0:
                    reasoning_effort = OpenAiReasoningEffort.LOW
                elif thinking_config.token_budget <= 4000:
                    reasoning_effort = OpenAiReasoningEffort.LOW
                elif thinking_config.token_budget <= 12000:
                    reasoning_effort = OpenAiReasoningEffort.MEDIUM
                else:
                    reasoning_effort = OpenAiReasoningEffort.HIGH

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logit_bias = {} if logit_bias is None else logit_bias
        self.completion = None
        self.reasoning_effort = reasoning_effort
        self.thinking_config = thinking_config

    def _convert_messages_to_responses_input(self, messages: List[Message], prompt_for_structured_output: bool = False) -> List[dict]:
        """Convert patterpunk messages to Responses API input format."""
        responses_input = []
        for message in messages:
            message_dict = message.to_dict(prompt_for_structured_output=prompt_for_structured_output)
            
            if message_dict["role"] == "system":
                responses_input.append({
                    "role": "developer",
                    "content": [{"type": "input_text", "text": message_dict["content"]}]
                })
            elif message_dict["role"] == "user":
                if isinstance(message_dict["content"], str):
                    responses_input.append({
                        "role": "user", 
                        "content": [{"type": "input_text", "text": message_dict["content"]}]
                    })
                else:
                    content_parts = []
                    for part in message_dict["content"]:
                        if part["type"] == "text":
                            content_parts.append({"type": "input_text", "text": part["text"]})
                        elif part["type"] == "image_url":
                            content_parts.append({"type": "input_image", "image_url": part["image_url"]["url"]})
                    responses_input.append({"role": "user", "content": content_parts})
            elif message_dict["role"] == "assistant":
                if message_dict.get("content"):
                    responses_input.append({
                        "role": "assistant",
                        "content": [{"type": "input_text", "text": message_dict["content"]}]
                    })
                if message_dict.get("tool_calls"):
                    responses_input.append({
                        "role": "assistant",
                        "content": [],
                        "tool_calls": message_dict["tool_calls"]
                    })
            elif message_dict["role"] == "tool_call":
                responses_input.append({
                    "role": "assistant",
                    "content": [],
                    "tool_calls": message_dict["tool_calls"]
                })
        
        return responses_input

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

        prompt_for_structured_output = False
        
        if structured_output and has_model_schema(structured_output):
            prompt_for_structured_output = True

        responses_input = self._convert_messages_to_responses_input(messages, prompt_for_structured_output)
        
        responses_parameters = {
            "model": self.model,
            "input": responses_input,
        }

        if tools:
            responses_parameters["tools"] = tools

        if structured_output and has_model_schema(structured_output) and not prompt_for_structured_output:
            responses_parameters["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": get_model_schema(structured_output),
                    "strict": True
                }
            }

        if self.model.startswith("o"):
            responses_parameters["reasoning"] = {
                "effort": self.reasoning_effort.name.lower()
            }
        else:
            responses_parameters["temperature"] = self.temperature
            responses_parameters["top_p"] = self.top_p
            if self.frequency_penalty != 0.0:
                responses_parameters["frequency_penalty"] = self.frequency_penalty
            if self.presence_penalty != 0.0:
                responses_parameters["presence_penalty"] = self.presence_penalty
            if self.logit_bias:
                responses_parameters["logit_bias"] = self.logit_bias

        log_params = {k: v for k, v in responses_parameters.items() if k != "input"}
        param_strings = []
        for key, value in log_params.items():
            param_strings.append(f"{key}: {value}")
        logger_llm.info(f"OpenAi Responses API params: {', '.join(param_strings)}")

        retry_count = 0
        done = False
        response = False

        while not done and retry_count < OPENAI_MAX_RETRIES:
            try:
                response = openai.responses.create(**responses_parameters)
                logger.info("OpenAi Responses API response received")
                done = True
            except APIError as error:
                if "reasoning.summary" in str(error) and "reasoning" in responses_parameters:
                    logger.info("Organization not verified for reasoning summaries, removing reasoning parameter and treating as regular model")
                    responses_parameters.pop("reasoning", None)
                    responses_parameters["temperature"] = self.temperature
                    responses_parameters["top_p"] = self.top_p
                else:
                    logger.info("Retrying OpenAi Responses API request due to APIError", exc_info=error)
                retry_count += 1

        if not done or not response:
            raise OpenAiApiError("OpenAi Responses API is returning too many api errors")

        logger_llm.info(f"[Assistant]\n{response.output_text}")

        if hasattr(response, 'output') and response.output:
            for output_item in response.output:
                if hasattr(output_item, 'tool_calls') and output_item.tool_calls:
                    tool_calls = []
                    for tool_call in output_item.tool_calls:
                        tool_calls.append({
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        })
                    return ToolCallMessage(tool_calls)

        parsed_output = None
        if structured_output and has_model_schema(structured_output):
            try:
                parsed_output = structured_output.model_validate_json(response.output_text)
            except Exception as e:
                logger.warning(f"Failed to parse structured output: {e}")
                try:
                    json_content = extract_json(response.output_text)
                    if json_content:
                        parsed_output = structured_output.model_validate(json_content)
                except Exception as fallback_error:
                    logger.warning(f"Fallback JSON parsing also failed: {fallback_error}")

        return AssistantMessage(
            response.output_text,
            structured_output=structured_output,
            parsed_output=parsed_output
        )

    @staticmethod
    def get_name():
        return "OpenAI"

    @staticmethod
    def get_available_models() -> List[str]:
        return [model.id for model in openai.models.list().data]
