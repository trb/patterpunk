import time
from abc import ABC
from typing import AsyncIterator, List, Optional, Set, Union

from patterpunk.config.defaults import (
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    RETRY_MIN_DELAY,
    RETRY_JITTER_FACTOR,
)
from patterpunk.config.providers.azure_openai import (
    azure_openai,
    azure_openai_async,
    azure_openai_reasoning,
    azure_openai_reasoning_async,
    AZURE_OPENAI_MAX_RETRIES,
)
from patterpunk.lib.retry import calculate_backoff_delay, extract_retry_after
from patterpunk.llm.models.openai import OpenAiModel, OpenAiApiError
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.types import ToolDefinition
from patterpunk.llm.messages.base import Message
from patterpunk.llm.streaming import StreamChunk, StreamEventType
from patterpunk.logger import logger, logger_llm

try:
    from openai import APIError, BadRequestError
except ImportError:
    pass  # openai package not installed, will fail at runtime if Azure OpenAI methods are called


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
            _INTERNAL__skip_client_validation=True,
        )

    def _execute_with_retry(self, responses_parameters: dict) -> object:
        # Use reasoning client for reasoning models, regular client otherwise
        if self._is_reasoning_model(self.model):
            if not azure_openai_reasoning:
                raise AzureOpenAiMissingConfigurationError(
                    "Azure OpenAI reasoning client was not initialized. "
                    "Check that PP_AZURE_OPENAI_REASONING_ENDPOINT and PP_AZURE_OPENAI_REASONING_API_KEY are set."
                )
            client = azure_openai_reasoning
        else:
            if not azure_openai:
                raise AzureOpenAiMissingConfigurationError(
                    "Azure OpenAI client was not initialized. "
                    "Check that PP_AZURE_OPENAI_ENDPOINT and PP_AZURE_OPENAI_API_KEY are set."
                )
            client = azure_openai

        retry_count = 0
        done = False
        response = False

        while not done and retry_count < AZURE_OPENAI_MAX_RETRIES:
            try:
                response = client.responses.create(**responses_parameters)
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
                    # Retry immediately for this specific error (no rate limit)
                    retry_count += 1
                    continue

                # Don't retry 400 errors - these are parameter errors, not transient failures
                if isinstance(error, BadRequestError):
                    logger.error(
                        f"Request failed with BadRequestError (not retrying): {error}"
                    )
                    raise

                # Calculate backoff delay for rate limit and transient errors
                retry_after = extract_retry_after(error)
                wait_time = calculate_backoff_delay(
                    attempt=retry_count,
                    base_delay=RETRY_BASE_DELAY,
                    max_delay=RETRY_MAX_DELAY,
                    min_delay=RETRY_MIN_DELAY,
                    jitter_factor=RETRY_JITTER_FACTOR,
                    retry_after=retry_after,
                )
                logger.warning(
                    f"Retrying request to /responses in {wait_time:.1f} seconds "
                    f"(attempt {retry_count + 1}/{AZURE_OPENAI_MAX_RETRIES})",
                    exc_info=error,
                )
                time.sleep(wait_time)
                retry_count += 1

        if not done or not response:
            raise OpenAiApiError(
                "Azure OpenAI Responses API is returning too many api errors"
            )

        return response

    def _log_request_start(self, messages: List) -> None:
        logger.info("Request to Azure OpenAI made")
        logger_llm.debug(
            "\n---\n".join(
                [f"{message.__repr__(truncate=False)}" for message in messages]
            )
        )

    def _log_request_parameters(self, responses_parameters: dict) -> None:
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

    def _is_reasoning_model(self, model: str) -> bool:
        """Check if this is a reasoning model that uses thinking tokens."""
        model_lower = model.lower()

        if model_lower.startswith(("o1", "o3")):
            return True

        if model_lower.startswith("gpt-"):
            try:
                # Extract major version (e.g., "5" from "gpt-5" or "gpt-5.2-turbo")
                version_str = model_lower[4:].split(".")[0].split("-")[0]
                major_version = int(version_str)
                return major_version >= 5
            except (ValueError, IndexError):
                return False

        return False

    def _setup_model_parameters(
        self,
        model: str,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        logit_bias: dict,
        reasoning_effort,
    ) -> dict:
        """
        Override to add summary='auto' for reasoning models.

        This enables streaming of thinking/reasoning summaries.
        """
        model_params = super()._setup_model_parameters(
            model,
            temperature,
            top_p,
            frequency_penalty,
            presence_penalty,
            logit_bias,
            reasoning_effort,
        )

        # Add summary='auto' to get reasoning summaries streamed
        if self._is_reasoning_model(model) and "reasoning" in model_params:
            model_params["reasoning"]["summary"] = "auto"

        return model_params

    async def stream_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream the assistant message response from Azure OpenAI.

        Yields StreamChunk objects for each streaming event.
        Uses the reasoning client for reasoning models (gpt-5.2, o1, o3-mini).
        """
        # Use reasoning client for reasoning models, regular client otherwise
        if self._is_reasoning_model(self.model):
            if not azure_openai_reasoning_async:
                raise AzureOpenAiMissingConfigurationError(
                    "Azure OpenAI reasoning async client was not initialized. "
                    "Check that PP_AZURE_OPENAI_API_KEY is set."
                )
            client = azure_openai_reasoning_async
        else:
            if not azure_openai_async:
                raise AzureOpenAiMissingConfigurationError(
                    "Azure OpenAI async client was not initialized. "
                    "Check that PP_AZURE_OPENAI_ENDPOINT and PP_AZURE_OPENAI_API_KEY are set."
                )
            client = azure_openai_async

        self._log_request_start(messages)

        responses_parameters = self._prepare_request_parameters(
            messages, tools, structured_output, output_types
        )
        responses_parameters["stream"] = True

        self._log_request_parameters(responses_parameters)

        async for event in self._stream_with_retry(
            client,
            responses_parameters,
            AZURE_OPENAI_MAX_RETRIES,
        ):
            chunk = self._convert_openai_event_to_chunk(event)
            if chunk is not None:
                yield chunk

    def _convert_openai_event_to_chunk(self, event) -> Optional[StreamChunk]:
        """
        Convert an OpenAI Responses API streaming event to a StreamChunk.

        Returns None for events we don't need to expose.
        """
        event_type = getattr(event, "type", None)

        # Reasoning/thinking content deltas (for o1, o3-mini, etc.)
        if event_type == "response.reasoning_summary_text.delta":
            return StreamChunk(
                event_type=StreamEventType.THINKING_DELTA,
                text=event.delta,
            )

        # Reasoning summary part lifecycle (block start)
        if event_type == "response.reasoning_summary_part.added":
            return StreamChunk(
                event_type=StreamEventType.CONTENT_BLOCK_START,
                block_type="thinking",
            )

        # Reasoning summary part done (block end)
        if event_type == "response.reasoning_summary_part.done":
            return StreamChunk(
                event_type=StreamEventType.CONTENT_BLOCK_STOP,
                block_type="thinking",
            )

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
            thinking_blocks = None
            response = getattr(event, "response", None)

            if response:
                if hasattr(response, "usage"):
                    usage_obj = response.usage
                    usage = {
                        "input_tokens": getattr(usage_obj, "input_tokens", 0),
                        "output_tokens": getattr(usage_obj, "output_tokens", 0),
                    }

                # Extract reasoning/thinking blocks from completed response
                thinking_blocks = self._extract_thinking_blocks_from_response(response)

            return StreamChunk(
                event_type=StreamEventType.MESSAGE_END,
                usage=usage,
                thinking_blocks=thinking_blocks,
            )

        return None
