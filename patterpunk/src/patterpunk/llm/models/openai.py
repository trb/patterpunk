import base64
import enum
from abc import ABC
from typing import List, Literal, Optional, Set, Union

from patterpunk.config.defaults import (
    DEFAULT_TEMPERATURE,
    GENERATE_STRUCTURED_OUTPUT_PROMPT,
)
from patterpunk.config.providers.openai import (
    openai,
    OPENAI_MAX_RETRIES,
)
from patterpunk.lib.structured_output import has_model_schema, get_model_schema
from patterpunk.lib.extract_json import extract_json
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.base import Message
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.models.base import Model
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.types import ToolDefinition, CacheChunk
from patterpunk.llm.chunks import MultimodalChunk, TextChunk
from patterpunk.llm.messages.cache import get_multimodal_chunks, has_multimodal_content
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

    def _process_cache_chunks_for_openai(
        self, chunks: List[Union[TextChunk, CacheChunk]]
    ) -> tuple[str, bool]:
        content = "".join(
            chunk.content
            for chunk in chunks
            if isinstance(chunk, (TextChunk, CacheChunk))
        )

        cacheable_positions = [
            i
            for i, chunk in enumerate(chunks)
            if isinstance(chunk, CacheChunk) and chunk.cacheable
        ]

        should_warn = False
        if cacheable_positions:
            last_cacheable = max(cacheable_positions)
            for i in range(last_cacheable):
                if not (isinstance(chunks[i], CacheChunk) and chunks[i].cacheable):
                    should_warn = True
                    break

        return content, should_warn

    def _convert_message_content_for_openai_responses(
        self, content, role: str
    ) -> List[dict]:
        # Assistant messages use "output_text" for text content
        # User/developer messages use "input_*" types
        # Note: Multimodal content in assistant messages (images/files) is not supported in input
        # as these are generated via tool calls (image_generation_call), not direct content

        if isinstance(content, str):
            text_type = "output_text" if role == "assistant" else "input_text"
            return [{"type": text_type, "text": content}]

        openai_content = []
        session = None

        for chunk in content:
            if isinstance(chunk, TextChunk):
                text_type = "output_text" if role == "assistant" else "input_text"
                openai_content.append({"type": text_type, "text": chunk.content})
            elif isinstance(chunk, CacheChunk):
                text_type = "output_text" if role == "assistant" else "input_text"
                openai_content.append({"type": text_type, "text": chunk.content})
            elif isinstance(chunk, MultimodalChunk):
                # Multimodal content only supported for non-assistant messages in input
                # Assistant-generated images/files come through tool calls in responses
                if role == "assistant":
                    logger.warning(
                        f"[OPENAI_RESPONSES_API] Skipping multimodal content in assistant message - "
                        f"images/files from assistant should be generated via tool calls, not direct content"
                    )
                    continue

                if chunk.media_type and chunk.media_type.startswith("image/"):
                    if chunk.source_type == "url":
                        openai_content.append(
                            {"type": "input_image", "image_url": chunk.get_url()}
                        )
                    else:
                        openai_content.append(
                            {"type": "input_image", "image_url": chunk.to_data_uri()}
                        )
                elif chunk.media_type == "application/pdf":
                    if chunk.source_type == "url":
                        openai_content.append(
                            {"type": "input_file", "file_url": chunk.get_url()}
                        )
                    elif hasattr(chunk, "file_id"):
                        openai_content.append(
                            {"type": "input_file", "file_id": chunk.file_id}
                        )
                    else:
                        openai_content.append(
                            {
                                "type": "input_file",
                                "filename": chunk.filename or "document.pdf",
                                "file_data": chunk.to_data_uri(),
                            }
                        )
                else:
                    openai_content.append(
                        {
                            "type": "input_file",
                            "filename": chunk.filename or "file",
                            "file_data": chunk.to_data_uri(),
                        }
                    )

        return openai_content

    def _convert_messages_for_openai_cache(self, messages: List[Message]) -> List[dict]:
        openai_messages = []
        cache_warnings = []

        for message in messages:
            if isinstance(message.content, list):
                content, should_warn = self._process_cache_chunks_for_openai(
                    message.content
                )
                if should_warn:
                    cache_warnings.append(
                        f"Non-prefix cacheable content detected in {message.role} message"
                    )
            else:
                content = message.content

            openai_messages.append({"role": message.role, "content": content})

        for warning in cache_warnings:
            logger.warning(f"[OPENAI_CACHE] {warning} - caching may be ineffective")

        return openai_messages

    def _has_corresponding_tool_result(
        self, messages: List[Message], tool_call_index: int
    ) -> bool:
        """
        Check if a ToolCallMessage has a corresponding ToolResultMessage.

        Scans messages after the tool_call_index to find ToolResultMessage(s)
        that match the call_ids from the ToolCallMessage.
        """
        tool_call_message = messages[tool_call_index]
        if not hasattr(tool_call_message, "tool_calls"):
            return False

        # Get all call_ids from this tool call message
        call_ids = {tc["id"] for tc in tool_call_message.tool_calls}

        # Look for corresponding ToolResultMessage(s) after this tool call
        for i in range(tool_call_index + 1, len(messages)):
            msg = messages[i]
            if hasattr(msg, "role") and msg.role == "tool_result":
                if hasattr(msg, "call_id") and msg.call_id in call_ids:
                    call_ids.remove(msg.call_id)
                    if not call_ids:  # All tool calls have results
                        return True

        return False

    def _convert_messages_to_responses_input(
        self, messages: List[Message], prompt_for_structured_output: bool = False
    ) -> List[dict]:
        responses_input = []
        cache_warnings = []

        for idx, message in enumerate(messages):
            if has_multimodal_content(message.content):
                content_array = self._convert_message_content_for_openai_responses(
                    message.content, message.role
                )
            else:
                if isinstance(message.content, list):
                    content_text, should_warn = self._process_cache_chunks_for_openai(
                        message.content
                    )
                    if should_warn:
                        cache_warnings.append(
                            f"Non-prefix cacheable content detected in {message.role} message"
                        )
                else:
                    content_text = message.get_content_as_string()

                if (
                    prompt_for_structured_output
                    and hasattr(message, "structured_output")
                    and message.structured_output
                    and has_model_schema(message.structured_output)
                ):
                    content_text = f"{content_text}\n{GENERATE_STRUCTURED_OUTPUT_PROMPT}{get_model_schema(message.structured_output)}"

                # Use "output_text" for assistant messages, "input_text" for others
                text_type = (
                    "output_text" if message.role == "assistant" else "input_text"
                )
                content_array = [{"type": text_type, "text": content_text}]

            if message.role == "system":
                responses_input.append({"role": "developer", "content": content_array})
            elif message.role == "user":
                responses_input.append({"role": "user", "content": content_array})
            elif message.role == "assistant":
                if content_array and any(
                    item.get("text")
                    for item in content_array
                    if item.get("type") in ["input_text", "output_text"]
                ):
                    responses_input.append(
                        {"role": "assistant", "content": content_array}
                    )
            elif message.role == "tool_call":
                # Only serialize tool calls if they have corresponding ToolResultMessage
                # The Responses API requires function_call to be paired with function_call_output
                # Orphaned tool calls (without results) are invalid and will be rejected by the API
                if self._has_corresponding_tool_result(messages, idx):
                    for tool_call in message.tool_calls:
                        responses_input.append(
                            {
                                "type": "function_call",
                                "call_id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            }
                        )
                else:
                    logger.warning(
                        f"[OPENAI] Skipping ToolCallMessage in conversation history - no corresponding ToolResultMessage found. "
                        f"Tool calls must be paired with results in the Responses API."
                    )
            elif message.role == "tool_result":
                # Validate required field for OpenAI
                if not message.call_id:
                    raise ValueError(
                        "OpenAI Responses API requires call_id in ToolResultMessage. "
                        "Ensure ToolResultMessage is created with call_id from the original ToolCallMessage."
                    )

                # Serialize as function_call_output
                # NOTE: This currently fails if the corresponding ToolCallMessage isn't in the input
                responses_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.call_id,
                        "output": message.content,
                    }
                )

        for warning in cache_warnings:
            logger.warning(f"[OPENAI_CACHE] {warning} - caching may be ineffective")

        return responses_input

    def _should_prompt_for_structured_output(
        self, structured_output: Optional[object]
    ) -> bool:
        prompt_for_structured_output = False
        if structured_output and has_model_schema(structured_output):
            prompt_for_structured_output = True
        return prompt_for_structured_output

    def _log_request_parameters(self, responses_parameters: dict) -> None:
        log_params = {k: v for k, v in responses_parameters.items() if k != "input"}
        param_strings = []
        for key, value in log_params.items():
            param_strings.append(f"{key}: {value}")
        logger_llm.info(f"OpenAi Responses API params: {', '.join(param_strings)}")

    def _is_reasoning_model(self, model: str) -> bool:
        return model.startswith("o")

    def _setup_tools_parameter(
        self,
        tools: Optional[ToolDefinition],
        output_types: Optional[Union[List[OutputType], Set[OutputType]]],
    ) -> Optional[List[dict]]:
        tools_list = None

        if tools:
            tools_list = tools if isinstance(tools, list) else [tools]
            # Convert from Chat Completions format to Responses API format
            tools_list = self._convert_tools_to_responses_format(tools_list)

        if output_types and OutputType.IMAGE in output_types:
            image_generation_tool = {"type": "image_generation"}
            if tools_list:
                tools_list = tools_list + [image_generation_tool]
            else:
                tools_list = [image_generation_tool]

        return tools_list

    def _convert_tools_to_responses_format(self, tools: List[dict]) -> List[dict]:
        """Convert tools from Chat Completions format to Responses API format.

        Chat Completions format: {"type": "function", "function": {"name": "...", ...}}
        Responses API format: {"type": "function", "name": "...", ...}
        """
        converted_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                # Flatten nested function structure
                flattened = {"type": "function"}
                flattened.update(tool["function"])
                converted_tools.append(flattened)
            else:
                # Keep non-function tools as-is
                converted_tools.append(tool)
        return converted_tools

    def _setup_structured_output_parameter(
        self, structured_output: Optional[object], prompt_for_structured_output: bool
    ) -> Optional[dict]:
        if (
            structured_output
            and has_model_schema(structured_output)
            and not prompt_for_structured_output
        ):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": get_model_schema(structured_output),
                    "strict": True,
                },
            }
        return None

    def _setup_model_parameters(
        self,
        model: str,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        logit_bias: dict,
        reasoning_effort: OpenAiReasoningEffort,
    ) -> dict:
        model_params = {}

        if self._is_reasoning_model(model):
            model_params["reasoning"] = {"effort": reasoning_effort.name.lower()}
        else:
            model_params["temperature"] = temperature
            model_params["top_p"] = top_p
            if frequency_penalty != 0.0:
                model_params["frequency_penalty"] = frequency_penalty
            if presence_penalty != 0.0:
                model_params["presence_penalty"] = presence_penalty
            if logit_bias:
                model_params["logit_bias"] = logit_bias

        return model_params

    def _process_image_generation_output(
        self, output_item
    ) -> Optional[MultimodalChunk]:
        if hasattr(output_item, "type") and output_item.type == "image_generation_call":
            if hasattr(output_item, "result") and output_item.result:
                image_chunk = MultimodalChunk.from_base64(
                    output_item.result, media_type="image/png"
                )
                return image_chunk
        return None

    def _process_tool_calls_output(self, output_item) -> List[dict]:
        tool_calls = []

        # Check if output_item IS a tool call (ResponseFunctionToolCall)
        if hasattr(output_item, "type") and output_item.type == "function_call":
            tool_calls.append(
                {
                    "id": output_item.call_id,
                    "type": "function",
                    "function": {
                        "name": output_item.name,
                        "arguments": output_item.arguments,
                    },
                }
            )
        # Legacy format: check if output_item has tool_calls attribute
        elif hasattr(output_item, "tool_calls") and output_item.tool_calls:
            for tool_call in output_item.tool_calls:
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
        return tool_calls

    def _parse_structured_output(
        self, response_text: str, structured_output: Optional[object]
    ) -> Optional[object]:
        parsed_output = None
        if structured_output and has_model_schema(structured_output):
            try:
                parsed_output = structured_output.model_validate_json(response_text)
            except Exception as e:
                logger.warning(f"Failed to parse structured output: {e}")
                try:
                    json_content = extract_json(response_text)
                    if json_content:
                        parsed_output = structured_output.model_validate(json_content)
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback JSON parsing also failed: {fallback_error}"
                    )
        return parsed_output

    def _process_response_output(
        self, response, structured_output: Optional[object]
    ) -> Union[AssistantMessage, ToolCallMessage, None]:
        if hasattr(response, "output") and response.output:
            chunks = []
            tool_calls = []

            for output_item in response.output:
                image_chunk = self._process_image_generation_output(output_item)
                if image_chunk:
                    chunks.append(image_chunk)
                else:
                    output_tool_calls = self._process_tool_calls_output(output_item)
                    tool_calls.extend(output_tool_calls)

            if tool_calls:
                return ToolCallMessage(tool_calls)

            if chunks:
                text_content = response.output_text if response.output_text else ""
                if text_content:
                    chunks.insert(0, TextChunk(text_content))
                return AssistantMessage(chunks, structured_output=structured_output)

        return None

    def _build_base_request_parameters(
        self, model: str, responses_input: List[dict]
    ) -> dict:
        return {
            "model": model,
            "input": responses_input,
            "store": False,
        }

    def _prepare_request_parameters(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition],
        structured_output: Optional[object],
        output_types: Optional[Union[List[OutputType], Set[OutputType]]],
    ) -> dict:
        prompt_for_structured_output = self._should_prompt_for_structured_output(
            structured_output
        )

        responses_input = self._convert_messages_to_responses_input(
            messages, prompt_for_structured_output
        )

        responses_parameters = self._build_base_request_parameters(
            self.model, responses_input
        )

        tools_parameter = self._setup_tools_parameter(tools, output_types)
        if tools_parameter:
            responses_parameters["tools"] = tools_parameter

        structured_output_parameter = self._setup_structured_output_parameter(
            structured_output, prompt_for_structured_output
        )
        if structured_output_parameter:
            responses_parameters["response_format"] = structured_output_parameter

        model_params = self._setup_model_parameters(
            self.model,
            self.temperature,
            self.top_p,
            self.frequency_penalty,
            self.presence_penalty,
            self.logit_bias,
            self.reasoning_effort,
        )
        responses_parameters.update(model_params)

        return responses_parameters

    def _execute_with_retry(self, responses_parameters: dict) -> object:
        retry_count = 0
        done = False
        response = False

        while not done and retry_count < OPENAI_MAX_RETRIES:
            try:
                response = openai.responses.create(**responses_parameters)
                logger.info("OpenAi Responses API response received")
                done = True
            except APIError as error:
                if (
                    "reasoning.summary" in str(error)
                    and "reasoning" in responses_parameters
                ):
                    logger.info(
                        "Organization not verified for reasoning summaries, removing reasoning parameter and treating as regular model"
                    )
                    responses_parameters.pop("reasoning", None)
                    responses_parameters["temperature"] = self.temperature
                    responses_parameters["top_p"] = self.top_p
                else:
                    logger.info(
                        "Retrying OpenAi Responses API request due to APIError",
                        exc_info=error,
                    )
                retry_count += 1

        if not done or not response:
            raise OpenAiApiError(
                "OpenAi Responses API is returning too many api errors"
            )

        return response

    def _process_response(
        self, response, structured_output: Optional[object]
    ) -> Union[AssistantMessage, ToolCallMessage]:
        logger_llm.info(f"[Assistant]\n{response.output_text}")

        response_message = self._process_response_output(response, structured_output)
        if response_message:
            return response_message

        parsed_output = self._parse_structured_output(
            response.output_text, structured_output
        )

        return AssistantMessage(
            response.output_text,
            structured_output=structured_output,
            parsed_output=parsed_output,
        )

    def _log_request_start(self, messages: List[Message]) -> None:
        logger.info("Request to OpenAi made")
        logger_llm.debug(
            "\n---\n".join(
                [f"{message.__repr__(truncate=False)}" for message in messages]
            )
        )

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> Union[AssistantMessage, ToolCallMessage]:
        self._log_request_start(messages)

        responses_parameters = self._prepare_request_parameters(
            messages, tools, structured_output, output_types
        )

        self._log_request_parameters(responses_parameters)

        response = self._execute_with_retry(responses_parameters)

        return self._process_response(response, structured_output)

    @staticmethod
    def get_name():
        return "OpenAI"

    @staticmethod
    def get_available_models() -> List[str]:
        return [model.id for model in openai.models.list().data]
