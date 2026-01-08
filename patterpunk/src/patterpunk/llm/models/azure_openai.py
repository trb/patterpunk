from abc import ABC
from typing import AsyncIterator, List, Optional, Set, Union

from patterpunk.config.providers.azure_openai import (
    azure_openai,
    azure_openai_async,
    AZURE_OPENAI_MAX_RETRIES,
)
from patterpunk.llm.models.openai import OpenAiModel, OpenAiApiError
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.types import ToolDefinition
from patterpunk.llm.messages.base import Message
from patterpunk.logger import logger

if azure_openai:
    from openai import APIError


class AzureOpenAiMissingConfigurationError(Exception):
    pass


class AzureOpenAiModel(OpenAiModel, ABC):
    def __init__(
        self,
        deployment_name="",
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        logit_bias=None,
        thinking_config: Optional[ThinkingConfig] = None,
    ):
        if not azure_openai:
            raise AzureOpenAiMissingConfigurationError(
                "Azure OpenAI was not initialized correctly. "
                "Check that PP_AZURE_OPENAI_ENDPOINT and PP_AZURE_OPENAI_API_KEY are set."
            )

        super().__init__(
            model=deployment_name,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            thinking_config=thinking_config,
        )

    def _execute_with_retry(self, responses_parameters: dict) -> object:
        retry_count = 0
        done = False
        response = False

        while not done and retry_count < AZURE_OPENAI_MAX_RETRIES:
            try:
                response = azure_openai.responses.create(**responses_parameters)
                logger.info("Azure OpenAI Responses API response received")
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
                        "Retrying Azure OpenAI Responses API request due to APIError",
                        exc_info=error,
                    )
                retry_count += 1

        if not done or not response:
            raise OpenAiApiError(
                "Azure OpenAI Responses API is returning too many api errors"
            )

        return response

    def _log_request_start(self, messages: List) -> None:
        logger.info("Request to Azure OpenAI made")
        from patterpunk.logger import logger_llm

        logger_llm.debug(
            "\n---\n".join(
                [f"{message.__repr__(truncate=False)}" for message in messages]
            )
        )

    def _log_request_parameters(self, responses_parameters: dict) -> None:
        from patterpunk.logger import logger_llm

        log_params = {k: v for k, v in responses_parameters.items() if k != "input"}
        param_strings = []
        for key, value in log_params.items():
            param_strings.append(f"{key}: {value}")
        logger_llm.info(
            f"Azure OpenAI Responses API params: {', '.join(param_strings)}"
        )

    @staticmethod
    def get_name():
        return "Azure OpenAI"

    @staticmethod
    def get_available_models() -> List[str]:
        # Azure uses deployment names which are user-specific
        # We cannot list them via API without additional permissions
        return []

    async def stream_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> AsyncIterator["StreamChunk"]:
        """
        Stream the assistant message response from Azure OpenAI.

        Yields StreamChunk objects for each streaming event.
        """
        from patterpunk.llm.streaming import StreamChunk, StreamEventType

        if not azure_openai_async:
            raise AzureOpenAiMissingConfigurationError(
                "Azure OpenAI async client was not initialized. "
                "Check that PP_AZURE_OPENAI_ENDPOINT and PP_AZURE_OPENAI_API_KEY are set."
            )

        self._log_request_start(messages)

        responses_parameters = self._prepare_request_parameters(
            messages, tools, structured_output, output_types
        )
        responses_parameters["stream"] = True

        self._log_request_parameters(responses_parameters)

        stream = await azure_openai_async.responses.create(**responses_parameters)

        async for event in stream:
            chunk = self._convert_openai_event_to_chunk(event)
            if chunk is not None:
                yield chunk

    def _convert_openai_event_to_chunk(self, event) -> Optional["StreamChunk"]:
        """
        Convert an OpenAI Responses API streaming event to a StreamChunk.

        Returns None for events we don't need to expose.
        """
        from patterpunk.llm.streaming import StreamChunk, StreamEventType

        event_type = getattr(event, "type", None)

        # Text content deltas
        if event_type == "response.output_text.delta":
            return StreamChunk(
                event_type=StreamEventType.TEXT_DELTA,
                text=event.delta,
            )

        # Function call (tool use) start - use output_item.added for function_call type
        if event_type == "response.output_item.added":
            item = event.item
            if getattr(item, "type", None) == "function_call":
                return StreamChunk(
                    event_type=StreamEventType.TOOL_USE_START,
                    tool_call_id=item.call_id,
                    tool_name=item.name,
                )

        # Function call arguments delta
        if event_type == "response.function_call_arguments.delta":
            return StreamChunk(
                event_type=StreamEventType.TOOL_USE_DELTA,
                tool_arguments_delta=event.delta,
            )

        # Function call done - marks end of tool use block
        if event_type == "response.function_call_arguments.done":
            return StreamChunk(
                event_type=StreamEventType.CONTENT_BLOCK_STOP,
            )

        # Output item done - can be tool call completion
        if event_type == "response.output_item.done":
            item = event.item
            if getattr(item, "type", None) == "function_call":
                return StreamChunk(
                    event_type=StreamEventType.TOOL_USE_STOP,
                    tool_call_id=getattr(item, "call_id", None),
                    tool_name=getattr(item, "name", None),
                )

        # Response completed - end of stream
        if event_type == "response.completed":
            usage = None
            response = getattr(event, "response", None)
            if response and hasattr(response, "usage"):
                usage_obj = response.usage
                usage = {
                    "input_tokens": getattr(usage_obj, "input_tokens", 0),
                    "output_tokens": getattr(usage_obj, "output_tokens", 0),
                }

            return StreamChunk(
                event_type=StreamEventType.MESSAGE_END,
                usage=usage,
            )

        return None
