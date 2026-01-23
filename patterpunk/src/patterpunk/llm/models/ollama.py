import json
import random
import time
from abc import ABC
from typing import List, Optional, Callable, Set, Union

from patterpunk.config.providers.ollama import (
    ollama,
    OLLAMA_MAX_RETRIES,
)
from patterpunk.config.defaults import (
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    RETRY_MIN_DELAY,
    RETRY_JITTER_FACTOR,
)
from patterpunk.lib.retry import calculate_backoff_delay
from patterpunk.logger import logger
from patterpunk.lib.structured_output import get_model_schema, has_model_schema
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.base import Message
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.models.base import Model
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.types import ToolDefinition, CacheChunk, ToolCall
from patterpunk.llm.chunks import MultimodalChunk, TextChunk
from patterpunk.llm.messages.cache import get_multimodal_chunks, has_multimodal_content


class OllamaAPIError(Exception):
    """Raised when Ollama API requests fail after all retry attempts."""

    pass


# Status codes that indicate transient errors and should trigger retry
RETRYABLE_STATUS_CODES = {429, 500, 503}
# Status codes that indicate client errors and should NOT be retried
NON_RETRYABLE_STATUS_CODES = {400, 404}


class OllamaModel(Model, ABC):
    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        num_ctx: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens

    def _prepare_tools(self, tools: ToolDefinition) -> List[dict]:
        ollama_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                ollama_tools.append(tool)

        return ollama_tools

    def _prepare_messages(
        self, messages: List[Message]
    ) -> tuple[List[dict], List[str]]:
        import tempfile
        import os

        ollama_messages = []
        all_images = []
        session = None
        temp_files = []

        try:
            for message in messages:
                if message.role == "tool_call":
                    # Serialize ToolCallMessage as assistant message with tool_calls (OpenAI-compatible)
                    # Convert ToolCall dataclasses to OpenAI format dicts
                    tool_calls_dict = [
                        tc.to_openai_format() for tc in message.tool_calls
                    ]
                    ollama_messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": tool_calls_dict,
                        }
                    )
                    continue

                if message.role == "tool_result":
                    # Validate required field for Ollama
                    if not message.call_id:
                        raise ValueError(
                            "Ollama requires call_id in ToolResultMessage. "
                            "Ensure ToolResultMessage is created with call_id from the original ToolCallMessage."
                        )

                    # Serialize as OpenAI-compatible tool message
                    ollama_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": message.call_id,
                            "content": message.content,
                        }
                    )
                    continue

                if isinstance(message.content, str):
                    content_text = message.content
                else:
                    text_parts = []
                    for chunk in message.content:
                        if isinstance(chunk, (TextChunk, CacheChunk)):
                            text_parts.append(chunk.content)
                    content_text = "".join(text_parts)

                message_images = []
                if isinstance(message.content, list):
                    for chunk in message.content:
                        if (
                            isinstance(chunk, MultimodalChunk)
                            and chunk.media_type
                            and chunk.media_type.startswith("image/")
                        ):

                            if chunk.source_type == "file_path":
                                message_images.append(str(chunk.get_file_path()))
                            else:
                                if chunk.source_type == "url":
                                    if session is None:
                                        try:
                                            import requests

                                            session = requests.Session()
                                        except ImportError:
                                            raise ImportError(
                                                "requests library required for URL support"
                                            )

                                    chunk = chunk.download(session)

                                media_type = chunk.media_type or "image/jpeg"
                                suffix = self._get_file_extension(media_type)

                                with tempfile.NamedTemporaryFile(
                                    suffix=suffix, delete=False
                                ) as tmp_file:
                                    tmp_file.write(chunk.to_bytes())
                                    temp_files.append(tmp_file.name)
                                    message_images.append(tmp_file.name)

                ollama_message = {"role": message.role, "content": content_text}

                if message_images:
                    ollama_message["images"] = message_images

                ollama_messages.append(ollama_message)
                all_images.extend(message_images)

            self._temp_files = temp_files

            return ollama_messages, all_images

        except Exception:
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            raise

    def _get_file_extension(self, media_type: str) -> str:
        extension_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
        }
        return extension_map.get(media_type, ".jpg")

    def _prepare_options(self) -> dict:
        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.presence_penalty is not None:
            options["repeat_penalty"] = self.presence_penalty
        if self.seed is not None:
            options["seed"] = self.seed
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        return options

    def _build_chat_params(
        self,
        ollama_messages: List[dict],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        stream: bool = False,
    ) -> dict:
        from patterpunk.lib.structured_output import get_model_schema, has_model_schema

        chat_params = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": stream,
            "format": (
                get_model_schema(structured_output)
                if has_model_schema(structured_output)
                else None
            ),
            "options": self._prepare_options(),
        }

        if tools:
            ollama_tools = self._prepare_tools(tools)
            if ollama_tools:
                chat_params["tools"] = ollama_tools

        return chat_params

    def _execute_chat_request(self, chat_params: dict) -> dict:
        from patterpunk.config.providers.ollama import ollama

        return ollama.chat(**chat_params)

    def _execute_with_retry(self, chat_params: dict) -> dict:
        """
        Execute chat request with retry logic for transient errors.

        Handles:
        - 503 Server Overloaded (queue full, resource exhaustion)
        - 429 Too Many Requests (proxy rate limits)
        - 500 Internal Server Error (transient model failures)
        - Connection errors (network failures, server restart)

        Uses exponential backoff with jitter (±50%) for consistent behavior
        across all providers. Minimum delay of 45s respects rate limit windows.

        Non-retryable:
        - 400 Bad Request (invalid parameters)
        - 404 Not Found (model doesn't exist)
        """
        try:
            from ollama import ResponseError
        except ImportError:
            ResponseError = None

        last_error = None

        for attempt in range(OLLAMA_MAX_RETRIES):
            try:
                return self._execute_chat_request(chat_params)

            except Exception as error:
                last_error = error

                # Check for Ollama-specific ResponseError with status codes
                if ResponseError is not None and isinstance(error, ResponseError):
                    status_code = getattr(error, "status_code", None)

                    if status_code in NON_RETRYABLE_STATUS_CODES:
                        raise

                    if status_code in RETRYABLE_STATUS_CODES:
                        if attempt < OLLAMA_MAX_RETRIES - 1:
                            # Calculate delay with exponential backoff and jitter
                            wait_time = calculate_backoff_delay(
                                attempt=attempt,
                                base_delay=RETRY_BASE_DELAY,
                                max_delay=RETRY_MAX_DELAY,
                                min_delay=RETRY_MIN_DELAY,
                                jitter_factor=RETRY_JITTER_FACTOR,
                            )

                            logger.warning(
                                f"Ollama request failed (attempt {attempt + 1}/{OLLAMA_MAX_RETRIES}): "
                                f"Status {status_code}. Retrying in {wait_time:.1f}s..."
                            )
                            time.sleep(wait_time)
                            continue
                        raise

                    # Unknown status code - don't retry
                    raise

                # Connection errors - always retry
                if "connection" in str(error).lower() or isinstance(
                    error, (ConnectionError, OSError)
                ):
                    if attempt < OLLAMA_MAX_RETRIES - 1:
                        # Calculate delay with exponential backoff and jitter
                        wait_time = calculate_backoff_delay(
                            attempt=attempt,
                            base_delay=RETRY_BASE_DELAY,
                            max_delay=RETRY_MAX_DELAY,
                            min_delay=RETRY_MIN_DELAY,
                            jitter_factor=RETRY_JITTER_FACTOR,
                        )

                        logger.warning(
                            f"Ollama connection failed (attempt {attempt + 1}/{OLLAMA_MAX_RETRIES}): "
                            f"{error}. Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    raise

                # Other exceptions - don't retry
                raise

        # All retries exhausted
        raise OllamaAPIError(
            f"Ollama API failed after {OLLAMA_MAX_RETRIES} retries"
        ) from last_error

    def _process_tool_calls(self, response: dict) -> Optional[ToolCallMessage]:
        if not response.get("message", {}).get("tool_calls"):
            return None

        tool_calls = []
        for tc in response["message"]["tool_calls"]:
            call_id = tc.get("id")
            if not call_id:
                call_id = f"call_{tc['function']['name']}_{random.randint(1000, 9999)}"

            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                    type=tc.get("type", "function"),
                )
            )

        if tool_calls:
            return ToolCallMessage(tool_calls)

        return None

    def _create_assistant_response(
        self, response: dict, structured_output: Optional[object] = None
    ) -> AssistantMessage:
        return AssistantMessage(
            response["message"]["content"], structured_output=structured_output
        )

    def _cleanup_temp_files(self):
        if hasattr(self, "_temp_files"):
            for temp_file in self._temp_files:
                try:
                    import os

                    os.unlink(temp_file)
                except:
                    pass
            del self._temp_files

    def _handle_response_errors(self, exception: Exception):
        self._cleanup_temp_files()
        raise exception

    def _validate_response(self, response: dict) -> bool:
        if not isinstance(response, dict):
            return False

        message = response.get("message")
        if not isinstance(message, dict):
            return False

        if "content" not in message and not message.get("tool_calls"):
            return False

        return True

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> Union[Message, "ToolCallMessage"]:
        ollama_messages, all_images = self._prepare_messages(messages)

        chat_params = self._build_chat_params(
            ollama_messages=ollama_messages,
            tools=tools,
            structured_output=structured_output,
            stream=False,
        )

        try:
            response = self._execute_with_retry(chat_params)
        except Exception as e:
            self._handle_response_errors(e)

        if not self._validate_response(response):
            self._handle_response_errors(
                ValueError(f"Invalid response format: {response}")
            )

        tool_call_response = self._process_tool_calls(response)
        if tool_call_response is not None:
            return tool_call_response

        self._cleanup_temp_files()
        return self._create_assistant_response(response, structured_output)

    @staticmethod
    def get_name():
        return "Ollama"

    @staticmethod
    def get_available_models() -> List[str]:
        return [model["model"] for model in ollama.list()["models"]]
