import random
import time
from abc import ABC
from typing import AsyncIterator, Callable, Dict, List, Optional, Set, Union

import tiktoken

from patterpunk.config.defaults import (
    GENERATE_STRUCTURED_OUTPUT_PROMPT,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_JITTER_FACTOR,
    RETRY_MAX_DELAY,
    RETRY_MIN_DELAY,
    SDK_MAX_RETRIES,
)
from patterpunk.config.providers.openai_compatible import (
    OPENAI_COMPATIBLE_DEFAULT_TIMEOUT,
)
from patterpunk.lib.extract_json import extract_json
from patterpunk.lib.retry import (
    calculate_backoff_delay,
    extract_retry_after,
    run_with_retry_config,
    stream_with_retry_config,
)
from patterpunk.lib.structured_output import get_model_schema, has_model_schema
from patterpunk.llm.chunks import CacheChunk, MultimodalChunk, TextChunk
from patterpunk.llm.finish_reason import FinishReason
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.base import Message
from patterpunk.llm.messages.provider_data import ProviderData
from patterpunk.llm.messages.roles import ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.models.base import Model
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.retry_config import RetryConfig
from patterpunk.llm.streaming import StreamChunk, StreamEventType
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.types import ToolCall, ToolDefinition
from patterpunk.logger import logger

try:
    from openai import OpenAI, AsyncOpenAI, APIError, BadRequestError
except ImportError:
    OpenAI = None
    AsyncOpenAI = None


class OpenAiCompatibleApiError(Exception):
    pass


_FINISH_REASON_MAP: dict = {
    "stop": FinishReason.STOP,
    "length": FinishReason.MAX_TOKENS,
    "tool_calls": FinishReason.TOOL_USE,
    "content_filter": FinishReason.SAFETY,
}


def _normalize_finish_reason(raw: Optional[str]) -> Optional[FinishReason]:
    if raw is None:
        return None
    return _FINISH_REASON_MAP.get(raw, FinishReason.OTHER)


def _generate_call_id(name: Optional[str]) -> str:
    return f"call_{name or 'tool'}_{random.randint(1000, 9999)}"


class OpenAiCompatibleModel(Model, ABC):
    """
    Chat-completions client for any OpenAI-compatible endpoint: vllm, ollama,
    llama.cpp, Vertex AI MaaS, Azure Foundry partner models. The Responses-API
    OpenAiModel cannot talk to these endpoints, which serve /chat/completions
    only.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        api_key_provider: Optional[Callable[[], str]] = None,
        default_headers: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        thinking_config: Optional[ThinkingConfig] = None,
        include_stream_usage: bool = True,
        timeout: int = OPENAI_COMPATIBLE_DEFAULT_TIMEOUT,
        retry_config: Optional[RetryConfig] = None,
    ):
        if OpenAI is None:
            raise ImportError(
                "The openai package is required for OpenAiCompatibleModel. "
                "Install it with: pip install openai"
            )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.api_key_provider = api_key_provider
        self.default_headers = default_headers
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.thinking_config = thinking_config
        self.include_stream_usage = include_stream_usage
        self.timeout = timeout
        self.retry_config = retry_config

        # The SDK refuses to build a client without an api_key, but local
        # endpoints (vllm, ollama, llama.cpp) accept any value. When an
        # api_key_provider is set, per-request Authorization headers override
        # this placeholder anyway.
        client_kwargs = {
            "base_url": base_url,
            "api_key": api_key or "unused",
            "default_headers": default_headers,
            "max_retries": SDK_MAX_RETRIES,
            "timeout": timeout,
        }
        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)

    def __deepcopy__(self, memo_dict):
        # Skip the SDK clients; their httpx connection pools hold RLocks that
        # cannot be pickled. The new instance rebuilds them in __init__.
        return OpenAiCompatibleModel(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            api_key_provider=self.api_key_provider,
            default_headers=self.default_headers,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            max_tokens=self.max_tokens,
            thinking_config=self.thinking_config,
            include_stream_usage=self.include_stream_usage,
            timeout=self.timeout,
            retry_config=self.retry_config,
        )

    def _request_headers(self) -> Optional[Dict[str, str]]:
        if self.api_key_provider is None:
            return None
        return {"Authorization": f"Bearer {self.api_key_provider()}"}

    def _is_openai_reasoning_family(self) -> bool:
        model = self.model.lower()
        if "-chat" in model:
            return False
        return model.startswith(("o1", "o3", "o4", "gpt-5"))

    def _resolve_reasoning_effort(self) -> str:
        if self.thinking_config.effort is not None:
            return self.thinking_config.effort
        budget = self.thinking_config.token_budget or 0
        if budget <= 4000:
            return "low"
        if budget <= 12000:
            return "medium"
        return "high"

    def _convert_content(self, content) -> Union[str, List[dict]]:
        if isinstance(content, str):
            return content
        parts = []
        for chunk in content:
            if isinstance(chunk, (TextChunk, CacheChunk)):
                parts.append({"type": "text", "text": chunk.content})
            elif isinstance(chunk, MultimodalChunk):
                media_type = chunk.media_type or ""
                if not media_type.startswith("image/"):
                    raise ValueError(
                        f"OpenAI-compatible endpoints support only text and image "
                        f"content; got media type '{media_type or 'unknown'}'."
                    )
                if chunk.source_type == "url":
                    url = chunk.source
                else:
                    url = f"data:{media_type};base64,{chunk.to_base64()}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
        if all(part["type"] == "text" for part in parts):
            return "".join(part["text"] for part in parts)
        return parts

    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        converted = []
        for message in messages:
            if message.role == "tool_call":
                converted.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            tool_call.to_openai_format()
                            for tool_call in message.tool_calls
                        ],
                    }
                )
            elif message.role == "tool_result":
                if not message.call_id:
                    raise ValueError(
                        "OpenAI-compatible endpoints require call_id in "
                        "ToolResultMessage. Ensure it is created with call_id "
                        "from the original ToolCallMessage."
                    )
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.call_id,
                        "content": message.content,
                    }
                )
            elif message.role in (ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT):
                converted.append(
                    {
                        "role": message.role,
                        "content": self._convert_content(message.content),
                    }
                )
        return converted

    def _build_request_params(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition],
        structured_output: Optional[object],
        stream: bool = False,
    ) -> dict:
        params = {
            "model": self.model,
            "messages": self._convert_messages(messages),
        }

        if self._is_openai_reasoning_family():
            self._warn_dropped_sampling_params()
            if self.max_tokens is not None:
                params["max_completion_tokens"] = self.max_tokens
            if self.thinking_config is not None:
                params["reasoning_effort"] = self._resolve_reasoning_effort()
        else:
            if self.thinking_config is not None:
                logger.warning(
                    f"[OPENAI_COMPATIBLE] '{self.model}' is not an OpenAI "
                    f"reasoning-family id; ignoring thinking_config."
                )
            for name, value in (
                ("temperature", self.temperature),
                ("top_p", self.top_p),
                ("frequency_penalty", self.frequency_penalty),
                ("presence_penalty", self.presence_penalty),
                ("max_tokens", self.max_tokens),
            ):
                if value is not None:
                    params[name] = value

        if tools:
            params["tools"] = tools

        if structured_output and has_model_schema(structured_output):
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": get_model_schema(structured_output),
                    "strict": True,
                },
            }

        if stream:
            params["stream"] = True
            if self.include_stream_usage:
                params["stream_options"] = {"include_usage": True}

        return params

    def _warn_dropped_sampling_params(self) -> None:
        dropped = [
            f"{name}={value}"
            for name, value in (
                ("temperature", self.temperature),
                ("top_p", self.top_p),
                ("frequency_penalty", self.frequency_penalty),
                ("presence_penalty", self.presence_penalty),
            )
            if value is not None
        ]
        if dropped:
            logger.warning(
                f"[OPENAI_COMPATIBLE] Reasoning model '{self.model}' rejects "
                f"sampling parameters. Dropping user-set value(s): "
                f"{', '.join(dropped)}. Use ThinkingConfig(effort=...) instead."
            )

    def _maybe_response_format_fallback(
        self, params: dict, error: Exception
    ) -> Optional[dict]:
        # Older vllm and some MaaS endpoints reject response_format json_schema
        # with a 400. One retry embeds the schema into the prompt instead;
        # extract_json recovers the object from the plain-text answer.
        if "response_format" not in params or "response_format" not in str(error):
            return None
        logger.warning(
            f"[OPENAI_COMPATIBLE] Endpoint rejected response_format for "
            f"'{self.model}'; retrying with prompt-based structured output."
        )
        schema = params["response_format"]["json_schema"]["schema"]
        fallback = {key: value for key, value in params.items() if key != "response_format"}
        instruction = f"{GENERATE_STRUCTURED_OUTPUT_PROMPT}{schema}"
        fallback_messages = list(fallback["messages"])
        last = fallback_messages[-1] if fallback_messages else None
        if last and last.get("role") == "user" and isinstance(last.get("content"), str):
            fallback_messages[-1] = {
                **last,
                "content": f"{last['content']}\n{instruction}",
            }
        else:
            fallback_messages.append({"role": "user", "content": instruction})
        fallback["messages"] = fallback_messages
        return fallback

    def _create_completion(self, params: dict):
        request_kwargs = dict(params)
        headers = self._request_headers()
        if headers:
            request_kwargs["extra_headers"] = headers
        return self._client.chat.completions.create(**request_kwargs)

    def _execute_with_retry(self, params: dict):
        if self.retry_config is not None:
            return run_with_retry_config(
                self.retry_config,
                lambda: self._create_completion(params),
                "OpenAI-compatible",
            )

        retry_count = 0
        while True:
            try:
                return self._create_completion(params)
            except BadRequestError:
                raise
            except APIError as error:
                if retry_count >= MAX_RETRIES:
                    raise OpenAiCompatibleApiError(
                        f"OpenAI-compatible endpoint {self.base_url} kept "
                        f"failing after {retry_count} retries"
                    ) from error
                wait_time = calculate_backoff_delay(
                    attempt=retry_count,
                    base_delay=RETRY_BASE_DELAY,
                    max_delay=RETRY_MAX_DELAY,
                    min_delay=RETRY_MIN_DELAY,
                    jitter_factor=RETRY_JITTER_FACTOR,
                    retry_after=extract_retry_after(error),
                )
                logger.warning(
                    f"[OPENAI_COMPATIBLE] Request failed "
                    f"(attempt {retry_count + 1}/{MAX_RETRIES}): {error}. "
                    f"Retrying in {wait_time:.1f}s."
                )
                time.sleep(wait_time)
                retry_count += 1

    def _parse_structured_output(
        self, response_text: str, structured_output: object
    ) -> Optional[object]:
        try:
            return structured_output.model_validate_json(response_text)
        except Exception as error:
            logger.warning(f"Failed to parse structured output: {error}")
            try:
                json_content = extract_json(response_text)
                if json_content:
                    return structured_output.model_validate(json_content)
            except Exception as fallback_error:
                logger.warning(f"Fallback JSON parsing also failed: {fallback_error}")
        return None

    def _process_response(
        self, response, structured_output: Optional[object]
    ) -> Union[AssistantMessage, ToolCallMessage]:
        choice = response.choices[0]
        message = choice.message
        raw_finish_reason = choice.finish_reason
        diagnostics_kwargs = {
            "finish_reason": _normalize_finish_reason(raw_finish_reason),
            "provider_data": ProviderData(raw_finish_reason=raw_finish_reason),
        }

        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tool_call.id or _generate_call_id(tool_call.function.name),
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments or "{}",
                )
                for tool_call in message.tool_calls
            ]
            return ToolCallMessage(tool_calls)

        content = message.content or ""

        thinking_blocks = None
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            thinking_blocks = [{"type": "thinking", "thinking": reasoning_content}]

        parsed_output = None
        if structured_output and has_model_schema(structured_output):
            parsed_output = self._parse_structured_output(content, structured_output)

        return AssistantMessage(
            content,
            structured_output=structured_output,
            parsed_output=parsed_output,
            thinking_blocks=thinking_blocks,
            **diagnostics_kwargs,
        )

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
        disable_safety_filters: bool = False,
    ) -> Union[Message, "ToolCallMessage"]:
        if disable_safety_filters:
            logger.debug(
                "[OPENAI_COMPATIBLE] disable_safety_filters has no effect — "
                "generic chat-completions endpoints expose no safety-filter "
                "parameters."
            )
        params = self._build_request_params(messages, tools, structured_output)

        try:
            response = self._execute_with_retry(params)
        except BadRequestError as error:
            fallback_params = self._maybe_response_format_fallback(params, error)
            if fallback_params is None:
                raise
            response = self._execute_with_retry(fallback_params)

        return self._process_response(response, structured_output)

    async def stream_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
        disable_safety_filters: bool = False,
    ) -> AsyncIterator["StreamChunk"]:
        if disable_safety_filters:
            logger.debug(
                "[OPENAI_COMPATIBLE] disable_safety_filters has no effect — "
                "generic chat-completions endpoints expose no safety-filter "
                "parameters."
            )
        params = self._build_request_params(
            messages, tools, structured_output, stream=True
        )

        if self.retry_config is not None:
            client = self._async_client.with_options(max_retries=0)

            async def acquire_stream():
                return self._stream_events(params, client)

            async for chunk in stream_with_retry_config(
                self.retry_config, acquire_stream, "OpenAI-compatible streaming"
            ):
                yield chunk
            return

        client = self._async_client.with_options(max_retries=MAX_RETRIES)
        async for chunk in self._stream_events(params, client):
            yield chunk

    async def _acquire_stream(self, params: dict, client):
        request_kwargs = dict(params)
        headers = self._request_headers()
        if headers:
            request_kwargs["extra_headers"] = headers
        return await client.chat.completions.create(**request_kwargs)

    async def _stream_events(
        self, params: dict, client
    ) -> AsyncIterator["StreamChunk"]:
        try:
            stream = await self._acquire_stream(params, client)
        except BadRequestError as error:
            fallback_params = self._maybe_response_format_fallback(params, error)
            if fallback_params is None:
                raise
            stream = await self._acquire_stream(fallback_params, client)

        async for chunk in self._iterate_stream(stream):
            yield chunk

    async def _iterate_stream(self, stream) -> AsyncIterator["StreamChunk"]:
        text_block_open = False
        thinking_block_open = False
        open_tool_index = None
        usage = None

        async for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = {
                    "input_tokens": getattr(chunk_usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(chunk_usage, "completion_tokens", 0)
                    or 0,
                }

            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta is not None:
                # reasoning_content is the deepseek/vllm extension for thinking
                # tokens; the upstream OpenAI schema does not define the field.
                reasoning_delta = getattr(delta, "reasoning_content", None)
                if reasoning_delta:
                    if not thinking_block_open:
                        yield StreamChunk(
                            event_type=StreamEventType.CONTENT_BLOCK_START,
                            block_type="thinking",
                        )
                        thinking_block_open = True
                    yield StreamChunk(
                        event_type=StreamEventType.THINKING_DELTA,
                        text=reasoning_delta,
                    )

                if delta.content:
                    if thinking_block_open:
                        yield StreamChunk(
                            event_type=StreamEventType.CONTENT_BLOCK_STOP
                        )
                        thinking_block_open = False
                    if not text_block_open:
                        yield StreamChunk(
                            event_type=StreamEventType.CONTENT_BLOCK_START,
                            block_type="text",
                        )
                        text_block_open = True
                    yield StreamChunk(
                        event_type=StreamEventType.TEXT_DELTA,
                        text=delta.content,
                    )

                for frame in delta.tool_calls or []:
                    frame_index = frame.index if frame.index is not None else 0
                    function = getattr(frame, "function", None)
                    if open_tool_index != frame_index:
                        if open_tool_index is not None:
                            yield StreamChunk(
                                event_type=StreamEventType.CONTENT_BLOCK_STOP,
                                index=open_tool_index,
                            )
                        name = function.name if function else None
                        yield StreamChunk(
                            event_type=StreamEventType.TOOL_USE_START,
                            index=frame_index,
                            tool_call_id=frame.id or _generate_call_id(name),
                            tool_name=name,
                        )
                        open_tool_index = frame_index
                    if function and function.arguments:
                        yield StreamChunk(
                            event_type=StreamEventType.TOOL_USE_DELTA,
                            tool_arguments_delta=function.arguments,
                            index=frame_index,
                        )

            if choice.finish_reason:
                if text_block_open:
                    yield StreamChunk(event_type=StreamEventType.CONTENT_BLOCK_STOP)
                    text_block_open = False
                if thinking_block_open:
                    yield StreamChunk(event_type=StreamEventType.CONTENT_BLOCK_STOP)
                    thinking_block_open = False
                if open_tool_index is not None:
                    yield StreamChunk(
                        event_type=StreamEventType.CONTENT_BLOCK_STOP,
                        index=open_tool_index,
                    )
                    open_tool_index = None

        # The usage-bearing chunk arrives after finish_reason, so MESSAGE_END
        # can only fire on stream exhaustion. ChatStream finalizes the message
        # and runs tool calls off this event.
        yield StreamChunk(event_type=StreamEventType.MESSAGE_END, usage=usage)

    @staticmethod
    def get_name():
        return "OpenAI Compatible"

    @staticmethod
    def get_available_models() -> List[str]:
        # Model listings are per endpoint; a static method cannot know which
        # base_url to ask. Query GET {base_url}/models directly if needed.
        return []

    def count_tokens(self, content: Union[str, Message, List[Message]]) -> int:
        """
        Approximate count using tiktoken's o200k_base encoding. Compatible
        endpoints serve arbitrary third-party tokenizers and expose no
        counting API, so expect deviations of 10-30 percent for non-OpenAI
        models. Good enough to drive context-window decisions.
        """
        encoding = tiktoken.get_encoding("o200k_base")
        if isinstance(content, str):
            return len(encoding.encode(content))
        if isinstance(content, list):
            return sum(self._count_message_tokens(m, encoding) for m in content)
        return self._count_message_tokens(content, encoding)

    def _count_message_tokens(self, message: Message, encoding) -> int:
        # Chat templates add about four structural tokens per message for
        # role markers and delimiters.
        total = 4
        text = (
            message.get_content_as_string()
            if hasattr(message, "get_content_as_string")
            else str(message.content)
        )
        total += len(encoding.encode(text))
        return total
