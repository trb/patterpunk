import asyncio
import json
import re
import time
from abc import ABC
from typing import AsyncIterator, Dict, List, Optional, Set, Tuple, Union

# Optional dependency for URL downloads in multimodal content
try:
    import requests as _requests_lib

    _requests_available = True
except ImportError:
    _requests_lib = None
    _requests_available = False

from patterpunk.config.defaults import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    GENERATE_STRUCTURED_OUTPUT_PROMPT,
    MAX_RETRIES,
    MIN_THINKING_BUDGET_TOKENS,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    RETRY_MIN_DELAY,
    RETRY_JITTER_FACTOR,
)
from patterpunk.lib.retry import (
    calculate_backoff_delay,
    run_with_retry_config,
    stream_with_retry_config,
)
from patterpunk.llm.retry_config import RetryConfig
from patterpunk.config.providers.bedrock import (
    boto3,
    get_bedrock_client_by_region,
    create_bedrock_client_for_streaming,
    BEDROCK_DEFAULT_TIMEOUT,
)
from patterpunk.lib.structured_output import get_model_schema, has_model_schema
from patterpunk.llm.models.claude_capabilities import (
    ClaudeVersion,
    SamplingParams,
    normalize_claude_model_id,
    parse_claude_version,
    resolve_claude_sampling,
    resolve_max_output_tokens,
    thinking_cannot_be_disabled,
)

if boto3:
    from botocore.exceptions import ClientError
from patterpunk.llm.finish_reason import FinishReason
from patterpunk.llm.messages.base import Message
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.provider_data import ProviderData
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.messages.roles import ROLE_SYSTEM, ROLE_ASSISTANT, ROLE_USER
from patterpunk.llm.models.base import Model, TokenCountingError
from patterpunk.llm.thinking import (
    EFFORT_TO_BUDGET,
    ThinkingConfig as UnifiedThinkingConfig,
    effort_for_budget,
)
from patterpunk.llm.types import ToolDefinition, CacheChunk, ToolCall
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.chunks import MultimodalChunk, TextChunk
from patterpunk.llm.messages.cache import get_multimodal_chunks, has_multimodal_content
from patterpunk.llm.streaming import StreamChunk, StreamEventType, StreamingError
from patterpunk.logger import logger, logger_llm

BEDROCK_STREAM_TIMEOUT_SECONDS = 300

THINKING_ANSWER_HEADROOM = 2000


class BedrockMissingCredentialsError(Exception):
    pass


# Native Bedrock Converse stopReason values → normalized FinishReason. Values
# not in this map (e.g. ``malformed_*``, ``model_context_window_exceeded``)
# fall through to OTHER.
_FINISH_REASON_MAP: dict = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.MAX_TOKENS,
    "tool_use": FinishReason.TOOL_USE,
    "guardrail_intervened": FinishReason.SAFETY,
    "content_filtered": FinishReason.SAFETY,
}


def _normalize_finish_reason(raw: Optional[str]) -> Optional[FinishReason]:
    if raw is None:
        return None
    return _FINISH_REASON_MAP.get(raw, FinishReason.OTHER)


def _build_diagnostics_kwargs(stop_reason: Optional[str]) -> dict:
    return {
        "finish_reason": _normalize_finish_reason(stop_reason),
        "provider_data": ProviderData(raw_finish_reason=stop_reason),
    }


def get_bedrock_conversation_content(message: Message):
    content_str = message.get_content_as_string()
    if message.structured_output and has_model_schema(message.structured_output):
        return f"{content_str}\n{GENERATE_STRUCTURED_OUTPUT_PROMPT}{get_model_schema(message.structured_output)}"

    return content_str


# A live Converse call to Claude 4.6 rejects "xhigh" with the error
# "Input should be 'low', 'medium', 'high' or 'max'". Sonnet 5 accepts it.
# Anthropic documents all five levels for 4.7+ and OpenAI for GPT-5.6+.
# AWS documents low/medium/high only for Nova 2 Lite.
_EFFORT_LEVELS_BASIC = ("low", "medium", "high")
_EFFORT_LEVELS_CLAUDE_4_6 = ("low", "medium", "high", "max")
_EFFORT_LEVELS_ALL = ("low", "medium", "high", "xhigh", "max")


def _is_openai_gpt5_model(model_id: str) -> bool:
    # normalize_claude_model_id strips the region prefix, ARN wrapper and ":0"
    # suffix that Bedrock wraps around every vendor's id, not only Anthropic's.
    # The gpt-oss ids stay outside this family: they are open-weight models
    # that accept sampling parameters and no reasoning field.
    return normalize_claude_model_id(model_id).startswith("openai.gpt-5")


def _is_nova_2_lite_model(model_id: str) -> bool:
    return "amazon.nova-2-lite" in model_id.lower()


class BedrockModel(Model, ABC):
    # Class-level tokenizer cache to avoid expensive reloads on each count_tokens() call
    # Loading HuggingFace tokenizers involves network calls and file parsing
    _llama_tokenizer_cache: Dict[str, "AutoTokenizer"] = {}
    _mistral_tokenizer: Optional["AutoTokenizer"] = None

    def __init__(
        self,
        model_id: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        thinking_config: Optional[UnifiedThinkingConfig] = None,
        timeout: int = BEDROCK_DEFAULT_TIMEOUT,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.thinking_config = thinking_config
        self.timeout = timeout
        self.retry_config = retry_config
        self._thinking_fields = self._resolve_thinking_fields()

        # When a RetryConfig governs retries, botocore's own retry layer must
        # not stack additional attempts underneath the configured schedule.
        sdk_max_attempts = 1 if retry_config is not None else None
        self.client = get_bedrock_client_by_region(
            client_type="bedrock-runtime",
            region=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            timeout=timeout,
            sdk_max_attempts=sdk_max_attempts,
        )

    def _convert_tools_to_bedrock_format(self, tools: ToolDefinition) -> dict:
        bedrock_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                bedrock_tool = {
                    "toolSpec": {
                        "name": func["name"],
                        "description": func["description"],
                        "inputSchema": {"json": func["parameters"]},
                    }
                }
                bedrock_tools.append(bedrock_tool)

        return {"tools": bedrock_tools}

    def _supports_reasoning_fields(self) -> bool:
        if _is_openai_gpt5_model(self.model_id) or _is_nova_2_lite_model(self.model_id):
            return True
        version = parse_claude_version(self.model_id)
        return version is not None and version.at_least(3, 7)

    def _resolve_thinking_fields(self) -> dict:
        config = self.thinking_config
        # DeepSeek reasons by default and takes no reasoning fields, so a
        # thinking_config is harmless there and stays silent.
        if config is None or "deepseek" in self.model_id.lower():
            return {}
        if not self._supports_reasoning_fields():
            logger.warning(
                f"[BEDROCK] Model '{self.model_id}' does not accept reasoning "
                f"parameters. Ignoring thinking_config."
            )
            return {}

        if _is_openai_gpt5_model(self.model_id):
            # GPT-5 reasons at effort "medium" when the request carries no
            # reasoning field. A zero budget therefore has to send "none"
            # explicitly instead of omitting the field like the other families.
            if config.token_budget == 0:
                return {"reasoning": {"effort": "none"}}
            return {"reasoning": {"effort": self._resolve_effort(_EFFORT_LEVELS_ALL)}}

        if _is_nova_2_lite_model(self.model_id):
            if config.token_budget == 0:
                return {}
            return {
                "reasoningConfig": {
                    "type": "enabled",
                    "maxReasoningEffort": self._resolve_effort(_EFFORT_LEVELS_BASIC),
                }
            }

        version = parse_claude_version(self.model_id)
        if not version.at_least(4, 6):
            if config.token_budget == 0:
                return {}
            return {
                "reasoning_config": {
                    "type": "enabled",
                    "budget_tokens": self._resolve_thinking_budget(),
                }
            }

        if config.token_budget == 0:
            return self._disabled_thinking_fields()
        # Sonnet 4.6 still accepts the enabled-with-budget shape, but
        # Anthropic deprecated it there, so 4.6 goes adaptive as well.
        if version.at_least(4, 7):
            return self._adaptive_thinking_fields(_EFFORT_LEVELS_ALL)
        return self._adaptive_thinking_fields(_EFFORT_LEVELS_CLAUDE_4_6)

    def _disabled_thinking_fields(self) -> dict:
        # Sonnet 5 and Opus 5 think by default, so omitting the field leaves
        # thinking on. Only an explicit "disabled" turns it off.
        if thinking_cannot_be_disabled(self.model_id):
            logger.warning(
                f"[BEDROCK] '{self.model_id}' cannot turn thinking off and "
                f"rejects thinking type 'disabled'. Ignoring token_budget=0; the "
                f"model runs adaptive thinking at its default effort. Pass "
                f"ThinkingConfig(effort='low') to keep thinking short instead."
            )
            return {}
        return {"reasoning_config": {"type": "disabled"}}

    def _adaptive_thinking_fields(self, accepted_levels: Tuple[str, ...]) -> dict:
        reasoning_config = {"type": "adaptive"}
        if self.thinking_config.include_thoughts:
            # Claude 4.7+ defaults display to "omitted" and returns empty
            # reasoning text unless the request asks for the summary.
            reasoning_config["display"] = "summarized"
        return {
            "reasoning_config": reasoning_config,
            "output_config": {"effort": self._resolve_effort(accepted_levels)},
        }

    def _resolve_effort(self, accepted_levels: Tuple[str, ...]) -> str:
        config = self.thinking_config
        if config.effort is None:
            effort = effort_for_budget(config.token_budget)
            logger.warning(
                f"[BEDROCK] '{self.model_id}' takes a reasoning effort level, not "
                f"a token budget. Coercing token_budget={config.token_budget} to "
                f"effort='{effort}'. Pass ThinkingConfig(effort='{effort}') "
                f"directly to silence this warning."
            )
            return effort
        if config.effort not in accepted_levels:
            logger.warning(
                f"[BEDROCK] '{self.model_id}' accepts effort levels "
                f"{'/'.join(accepted_levels)} only. Clamping effort="
                f"'{config.effort}' to 'high'."
            )
            return "high"
        return config.effort

    def _resolve_thinking_budget(self) -> int:
        config = self.thinking_config
        if config.effort is not None:
            return EFFORT_TO_BUDGET[self._resolve_effort(_EFFORT_LEVELS_BASIC)]
        if config.token_budget < MIN_THINKING_BUDGET_TOKENS:
            logger.warning(
                f"[BEDROCK] The API requires a thinking budget of at least "
                f"{MIN_THINKING_BUDGET_TOKENS} tokens. Raising "
                f"token_budget={config.token_budget} to {MIN_THINKING_BUDGET_TOKENS}."
            )
            return MIN_THINKING_BUDGET_TOKENS
        cap = self._output_token_cap()
        if cap is not None and config.token_budget + THINKING_ANSWER_HEADROOM > cap:
            budget = cap - THINKING_ANSWER_HEADROOM
            logger.warning(
                f"[BEDROCK] The thinking budget must stay below maxTokens, and "
                f"'{self.model_id}' caps output at {cap} tokens. Lowering the "
                f"thinking budget to {budget}."
            )
            return budget
        return config.token_budget

    def _get_thinking_params(self) -> dict:
        return self._thinking_fields

    def _convert_content_to_bedrock_format(self, content) -> List[dict]:
        if isinstance(content, str):
            return [{"text": content}]

        bedrock_content = []
        session = None

        for chunk in content:
            if isinstance(chunk, TextChunk):
                bedrock_content.append({"text": chunk.content})

            elif isinstance(chunk, CacheChunk):
                # Converse requires cachePoint as its own content block with a
                # "type" field. Embedding it in the text block is rejected
                # with "ValidationException: ... type: Field required".
                bedrock_content.append({"text": chunk.content})
                if chunk.cacheable:
                    bedrock_content.append({"cachePoint": {"type": "default"}})

            elif isinstance(chunk, MultimodalChunk):
                if chunk.source_type == "url":
                    if not _requests_available:
                        raise ImportError(
                            "requests library required for URL support with Bedrock. "
                            "Install with: pip install requests"
                        )
                    if session is None:
                        session = _requests_lib.Session()
                    chunk = chunk.download(session)

                media_type = chunk.media_type or "application/octet-stream"

                if media_type.startswith("image/"):
                    format_map = {
                        "image/jpeg": "jpeg",
                        "image/jpg": "jpeg",
                        "image/png": "png",
                        "image/gif": "gif",
                        "image/webp": "webp",
                    }

                    format = format_map.get(media_type, "jpeg")

                    content_block = {
                        "image": {
                            "format": format,
                            "source": {"bytes": chunk.to_bytes()},
                        }
                    }
                    bedrock_content.append(content_block)
                elif media_type in [
                    "application/pdf",
                    "text/plain",
                    "text/html",
                    "text/markdown",
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ]:
                    format_map = {
                        "application/pdf": "pdf",
                        "text/plain": "txt",
                        "text/html": "html",
                        "text/markdown": "md",
                        "application/msword": "doc",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                        "application/vnd.ms-excel": "xls",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
                    }

                    document_format = format_map.get(media_type, "pdf")
                    raw_filename = (
                        getattr(chunk, "filename", None)
                        or f"document.{document_format}"
                    )
                    document_name = re.sub(r"[^\w\s\-\(\)\[\]]", "", raw_filename)
                    document_name = re.sub(r"\s+", " ", document_name).strip()

                    content_block = {
                        "document": {
                            "format": document_format,
                            "name": document_name,
                            "source": {"bytes": chunk.to_bytes()},
                        }
                    }
                    bedrock_content.append(content_block)
                else:
                    raise ValueError(
                        f"Bedrock does not support media type: {media_type}. "
                        f"Supported types are: text, images (jpeg/png/gif/webp), and documents (pdf/csv/doc/docx/xls/xlsx/html/txt/md)."
                    )

        return bedrock_content

    def _convert_messages_for_bedrock(self, messages: List[Message]) -> List[dict]:
        """Convert patterpunk messages to Bedrock converse format."""
        bedrock_messages = []

        for message in messages:
            if message.role == "tool_call":
                bedrock_messages.append(self._convert_tool_call_message(message))
            elif message.role == "tool_result":
                self._append_tool_result(message, bedrock_messages)
            else:
                bedrock_messages.append(self._convert_regular_message(message))

        return bedrock_messages

    def _convert_tool_call_message(self, message) -> dict:
        """Convert ToolCallMessage to Bedrock assistant message with toolUse blocks.

        When extended thinking is enabled, thinking blocks must come FIRST
        in the content array (required by Bedrock).
        """
        content_blocks = []
        content_blocks.extend(self._extract_thinking_content_blocks(message))
        content_blocks.extend(self._extract_tool_use_blocks(message))
        return {"role": "assistant", "content": content_blocks}

    def _extract_thinking_content_blocks(self, message) -> List[dict]:
        """Extract thinking blocks for Bedrock format (must come first in content).

        Bedrock expects: {"reasoningContent": {"reasoningText": {"text": "...", "signature": "..."}}}
        """
        blocks = []
        if not hasattr(message, "thinking_blocks") or not message.thinking_blocks:
            return blocks

        for block in message.thinking_blocks:
            thinking_text = block.get("thinking", "")
            signature = block.get("signature", "")
            if block.get("type") == "thinking" and (thinking_text or signature):
                blocks.append(
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": thinking_text,
                                "signature": signature,
                            }
                        }
                    }
                )
        return blocks

    def _extract_tool_use_blocks(self, message) -> List[dict]:
        """Extract tool use blocks from ToolCallMessage."""
        blocks = []
        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Failed to parse tool arguments for '{tool_call.name}': {e}. "
                    f"Arguments: {tool_call.arguments[:100] if tool_call.arguments else 'empty'}"
                )
                arguments = {}

            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": tool_call.id,
                        "name": tool_call.name,
                        "input": arguments,
                    }
                }
            )
        return blocks

    def _append_tool_result(self, message, bedrock_messages: List[dict]) -> None:
        """Append tool result, merging with previous user message if needed.

        Bedrock requires consecutive tool results to be in a SINGLE user message
        with multiple toolResult blocks, not separate user messages.
        """
        if not message.call_id:
            raise ValueError(
                "AWS Bedrock requires call_id (as toolUseId) in ToolResultMessage. "
                "Ensure ToolResultMessage is created with call_id from the original ToolCallMessage."
            )

        tool_result_block = {
            "toolResult": {
                "toolUseId": message.call_id,
                "content": [{"text": message.content}],
                "status": "error" if message.is_error else "success",
            }
        }

        # Check if previous message is a user message with toolResult blocks
        should_merge = (
            bedrock_messages
            and bedrock_messages[-1]["role"] == "user"
            and bedrock_messages[-1]["content"]
            and "toolResult" in bedrock_messages[-1]["content"][0]
        )

        if should_merge:
            bedrock_messages[-1]["content"].append(tool_result_block)
        else:
            bedrock_messages.append({"role": "user", "content": [tool_result_block]})

    def _convert_regular_message(self, message: Message) -> dict:
        """Convert regular user/assistant message to Bedrock format."""
        if isinstance(message.content, list):
            content = self._convert_content_to_bedrock_format(message.content)

            has_text = any("text" in chunk for chunk in content)
            has_document = any("document" in chunk for chunk in content)

            if has_document and not has_text:
                raise ValueError(
                    "Bedrock requires at least one text block when documents are present. "
                    "Please include text content along with your documents."
                )
        else:
            content_str = message.get_content_as_string()
            if message.structured_output and has_model_schema(
                message.structured_output
            ):
                content_str = f"{content_str}\n{GENERATE_STRUCTURED_OUTPUT_PROMPT}{get_model_schema(message.structured_output)}"
            content = [{"text": content_str}]

        return {"role": message.role, "content": content}

    def _convert_system_messages_for_bedrock(
        self, messages: List[Message]
    ) -> List[dict]:
        bedrock_system = []

        for message in messages:
            if message.role == ROLE_SYSTEM:
                if isinstance(message.content, list):
                    content_str = "".join(chunk.content for chunk in message.content)
                    bedrock_system.append({"text": content_str})
                else:
                    bedrock_system.append({"text": message.content})

        return bedrock_system

    def _prepare_messages_for_converse(
        self, messages: List[Message]
    ) -> tuple[List[dict], Optional[List[dict]]]:
        system_messages = [
            message for message in messages if message.role == ROLE_SYSTEM
        ]
        system_content = self._convert_system_messages_for_bedrock(system_messages)

        user_assistant_messages = [
            message
            for message in messages
            if message.role in [ROLE_USER, ROLE_ASSISTANT, "tool_call", "tool_result"]
        ]
        conversation = self._convert_messages_for_bedrock(user_assistant_messages)

        return conversation, system_content if system_content else None

    def _build_claude_sampling_config(self, version: ClaudeVersion) -> dict:
        if not version.recognized:
            logger.warning(
                f"[BEDROCK] Unrecognised Claude model id '{self.model_id}'. "
                f"Treating it as a Claude 5 model."
            )

        reasoning_type = self._thinking_fields.get("reasoning_config", {}).get("type")
        resolution = resolve_claude_sampling(
            version,
            thinking_enabled=reasoning_type in ("enabled", "adaptive"),
            requested=SamplingParams(temperature=self.temperature, top_p=self.top_p),
            defaults=SamplingParams(temperature=DEFAULT_TEMPERATURE),
        )
        for warning in resolution.warnings:
            logger.warning(f"[BEDROCK] {warning}")

        sampling_config = {}
        if resolution.temperature is not None:
            sampling_config["temperature"] = resolution.temperature
        if resolution.top_p is not None:
            sampling_config["topP"] = resolution.top_p
        return sampling_config

    def _build_gpt5_sampling_config(self) -> dict:
        dropped = []
        if self.temperature != DEFAULT_TEMPERATURE:
            dropped.append(f"temperature={self.temperature}")
        if self.top_p is not None:
            dropped.append(f"top_p={self.top_p}")
        if dropped:
            logger.warning(
                f"[BEDROCK] OpenAI GPT-5 models reject sampling parameters. "
                f"Dropping user-set value(s): {', '.join(dropped)}. "
                f"Use ThinkingConfig(effort=...) to control output instead."
            )
        return {}

    def _build_nova_high_effort_config(self) -> dict:
        # Nova 2 Lite rejects temperature, topP, topK and maxTokens once
        # maxReasoningEffort is "high"; the model sizes its own output then.
        # https://docs.aws.amazon.com/nova/latest/nova2-userguide/extended-thinking.html
        dropped = []
        if self.temperature != DEFAULT_TEMPERATURE:
            dropped.append(f"temperature={self.temperature}")
        if self.top_p is not None:
            dropped.append(f"top_p={self.top_p}")
        if self.max_tokens:
            dropped.append(f"max_tokens={self.max_tokens}")
        if dropped:
            logger.warning(
                f"[BEDROCK] Nova 2 rejects sampling parameters and maxTokens at "
                f"effort='high'. Dropping user-set value(s): {', '.join(dropped)}."
            )
        return {}

    def _nova_reasoning_effort(self) -> Optional[str]:
        return self._thinking_fields.get("reasoningConfig", {}).get(
            "maxReasoningEffort"
        )

    def _build_inference_config(
        self, structured_output: Optional[object] = None
    ) -> dict:
        if self._nova_reasoning_effort() == "high":
            return self._build_nova_high_effort_config()

        # Anthropic enforces its sampling rules through Bedrock unchanged, so
        # Claude ids go through the shared resolver. GPT-5 ids reject sampling
        # outright. Other model families (Llama, Mistral, Nova, Titan, DeepSeek,
        # gpt-oss) accept temperature and topP.
        claude_version = parse_claude_version(self.model_id)
        if claude_version is not None:
            inference_config = self._build_claude_sampling_config(claude_version)
        elif _is_openai_gpt5_model(self.model_id):
            inference_config = self._build_gpt5_sampling_config()
        else:
            inference_config = {"temperature": self.temperature}
            if self.top_p is not None:
                inference_config["topP"] = self.top_p

        max_tokens = self._resolve_max_tokens()
        if max_tokens:
            inference_config["maxTokens"] = max_tokens

        return inference_config

    def _output_token_cap(self) -> Optional[int]:
        # Bedrock answers a maxTokens above the model's ceiling with a
        # ValidationException naming the limit. Live checks: GPT-5.6 Luna
        # 131072, Nova 2 Lite 65535, Claude 4.5 64000, Claude 4.6+ 128000.
        # Claude 3.7 keeps the 131072 ceiling recorded before these checks;
        # the direct Messages API caps that model at 64000.
        if _is_openai_gpt5_model(self.model_id):
            return 131072
        if _is_nova_2_lite_model(self.model_id):
            return 65535
        version = parse_claude_version(self.model_id)
        if version is None:
            return None
        if version.recognized and (version.major, version.minor) == (3, 7):
            return 131072
        return resolve_max_output_tokens(version, self.model_id)

    def _resolve_max_tokens(self) -> Optional[int]:
        max_tokens = self.max_tokens
        thinking_budget = self._thinking_fields.get("reasoning_config", {}).get(
            "budget_tokens"
        )
        # Anthropic returns a ValidationException when max_tokens is not above
        # budget_tokens. The headroom leaves room for the answer and matches
        # the margin the direct Anthropic model applies.
        if thinking_budget is not None and (
            not max_tokens or max_tokens <= thinking_budget
        ):
            max_tokens = thinking_budget + THINKING_ANSWER_HEADROOM

        cap = self._output_token_cap()
        if max_tokens and cap is not None and max_tokens > cap:
            logger.warning(
                f"[BEDROCK] max_tokens={max_tokens} exceeds the {cap}-token "
                f"output limit of '{self.model_id}'. Lowering maxTokens to {cap}."
            )
            return cap
        return max_tokens

    def _prepare_tool_config(self, tools: Optional[ToolDefinition]) -> Optional[dict]:
        if not tools:
            return None

        tool_config = self._convert_tools_to_bedrock_format(tools)
        if tool_config["tools"]:
            return tool_config
        return None

    def _build_converse_params(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
    ) -> dict:
        conversation, system_content = self._prepare_messages_for_converse(messages)
        inference_config = self._build_inference_config(structured_output)

        converse_params = {
            "modelId": self.model_id,
            "messages": conversation,
        }
        if inference_config:
            converse_params["inferenceConfig"] = inference_config

        if system_content:
            converse_params["system"] = system_content

        thinking_params = self._get_thinking_params()
        if thinking_params:
            converse_params["additionalModelRequestFields"] = thinking_params

        tool_config = self._prepare_tool_config(tools)
        if tool_config:
            converse_params["toolConfig"] = tool_config

        return self._enforce_cache_point_rules(converse_params)

    def _supports_cache_points(self) -> bool:
        # Bedrock's prompt-caching allowlist is narrower than Anthropic's own
        # API: Claude 3.7+, Claude 3.5 Haiku, and the Amazon Nova family.
        # Claude 3.5 Sonnet stayed preview-only and never reached GA.
        # Unsupported models reject cachePoint blocks with ValidationException.
        model_lower = self.model_id.lower()
        if "amazon.nova" in model_lower:
            return True
        version = parse_claude_version(self.model_id)
        if version is None:
            return False
        if version.at_least(3, 7):
            return True
        return (version.major, version.minor) == (3, 5) and "haiku" in (
            normalize_claude_model_id(self.model_id)
        )

    def _collect_cache_points(self, converse_params: dict) -> List[tuple]:
        containers = []
        system = converse_params.get("system")
        if isinstance(system, list):
            containers.append(system)
        for message in converse_params.get("messages", []):
            content = message.get("content")
            if isinstance(content, list):
                containers.append(content)

        return [
            (container, block)
            for container in containers
            for block in container
            if isinstance(block, dict) and "cachePoint" in block
        ]

    def _enforce_cache_point_rules(self, converse_params: dict) -> dict:
        def remove_block(container, block):
            for index, candidate in enumerate(container):
                if candidate is block:
                    del container[index]
                    return

        cache_points = self._collect_cache_points(converse_params)
        if not cache_points:
            return converse_params

        if not self._supports_cache_points():
            for container, block in cache_points:
                remove_block(container, block)
            logger.warning(
                f"[BEDROCK] Model '{self.model_id}' does not support prompt "
                f"caching on Bedrock. Removed {len(cache_points)} cache "
                f"checkpoint(s); the content is sent uncached."
            )
            return converse_params

        if len(cache_points) > 4:
            removed_count = len(cache_points) - 4
            for container, block in cache_points[:-4]:
                remove_block(container, block)
            logger.warning(
                f"[BEDROCK] Bedrock allows at most 4 cache checkpoints per "
                f"request. Removed the first {removed_count}; each remaining "
                f"checkpoint still caches all content before it."
            )
        return converse_params

    def _process_converse_response(
        self,
        output: dict,
        structured_output: Optional[object] = None,
        stop_reason: Optional[str] = None,
    ) -> Union[AssistantMessage, ToolCallMessage]:
        # Check for tool use in content blocks regardless of stopReason
        # Some models return tool calls in content without setting stopReason to "tool_use"
        tool_calls = []
        for content_block in output.get("message", {}).get("content", []):
            if "toolUse" in content_block:
                tool_use = content_block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        id=tool_use["toolUseId"],
                        name=tool_use["name"],
                        arguments=json.dumps(tool_use["input"]),
                    )
                )

        if tool_calls:
            return ToolCallMessage(tool_calls)

        response_content = output["message"]["content"]
        response_text = ""
        reasoning_text = ""

        for content_block in response_content:
            if "text" in content_block:
                response_text += content_block["text"]
            elif "reasoningContent" in content_block:
                reasoning_content = content_block["reasoningContent"]
                if "reasoningText" in reasoning_content:
                    reasoning_text += reasoning_content["reasoningText"]["text"]

        if (
            reasoning_text
            and self.thinking_config
            and self.thinking_config.include_thoughts
        ):
            full_response = (
                f"<thinking>\n{reasoning_text}\n</thinking>\n\n{response_text}"
            )
        else:
            full_response = response_text

        logger_llm.info(f"[Assistant]\n{full_response}")

        return AssistantMessage(
            full_response,
            structured_output=structured_output,
            **_build_diagnostics_kwargs(stop_reason),
        )

    # Error codes that indicate transient rate limiting and should trigger retry
    RETRYABLE_ERROR_CODES = ("ThrottlingException", "ServiceUnavailableException")

    def _execute_with_retry(self, operation, operation_name: str = "Bedrock API call"):
        """
        Execute an operation with exponential backoff retry on transient errors.

        Retries on:
        - ThrottlingException (429): Account quotas exceeded
        - ServiceUnavailableException (503): Service temporarily unavailable

        Uses exponential backoff with jitter (±50%) for consistent behavior
        across all providers. Minimum delay of 45s respects rate limit windows.

        Args:
            operation: Callable that performs the API operation
            operation_name: Name for logging purposes

        Returns:
            The result of the operation

        Raises:
            ClientError: If non-retryable error or max retries exceeded
        """
        if self.retry_config is not None:
            return run_with_retry_config(self.retry_config, operation, operation_name)

        retry = 0

        while True:
            try:
                return operation()
            except ClientError as client_exception:
                error_code = client_exception.response["Error"]["Code"]
                if error_code in self.RETRYABLE_ERROR_CODES:
                    if retry >= MAX_RETRIES:
                        logger.error(
                            f"ERROR: {operation_name} failed with {error_code}, "
                            f"max retries ({MAX_RETRIES}) reached"
                        )
                        raise

                    # Calculate delay with exponential backoff and jitter
                    wait_time = calculate_backoff_delay(
                        attempt=retry,
                        base_delay=RETRY_BASE_DELAY,
                        max_delay=RETRY_MAX_DELAY,
                        min_delay=RETRY_MIN_DELAY,
                        jitter_factor=RETRY_JITTER_FACTOR,
                    )

                    logger.warning(
                        f"{operation_name} received {error_code}, "
                        f"backing off ({wait_time:.1f}s) and retrying"
                    )
                    retry += 1
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"{operation_name} client exception: {error_code}",
                        exc_info=client_exception,
                    )
                    raise

    async def _stream_with_retry(
        self,
        streaming_client,
        converse_params: dict,
        max_retries: int,
    ) -> AsyncIterator["StreamChunk"]:
        """
        Execute streaming request with retry logic for pre-stream API errors.

        Retries on ThrottlingException and ServiceUnavailableException that
        occur BEFORE streaming begins. Once the EventStream starts yielding
        events, errors cannot be retried without data loss.

        Uses exponential backoff with jitter (±50%) for consistent behavior
        across all providers. Minimum delay of 45s respects rate limit windows.

        Note: Mid-stream throttling errors (wrapped in EventStreamError) will
        propagate to caller as they cannot be safely retried.

        Args:
            streaming_client: The boto3 bedrock-runtime client for streaming
            converse_params: Parameters for converse_stream API call
            max_retries: Maximum number of retry attempts

        Yields:
            StreamChunk objects from the stream

        Raises:
            ClientError: If non-retryable error or max retries exceeded
        """
        retry = 0

        while retry <= max_retries:
            try:
                # Attempt to initiate stream
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: streaming_client.converse_stream(**converse_params),
                )

                stream = response.get("stream")
                if not stream:
                    raise ValueError("No stream returned from converse_stream")

                # Stream initiated successfully - yield all events
                async for chunk in self._iterate_stream_events(stream):
                    yield chunk

                return  # Success - exit retry loop

            except ClientError as error:
                error_code = error.response["Error"]["Code"]

                # Only retry on specific transient errors
                if error_code not in self.RETRYABLE_ERROR_CODES:
                    logger.error(
                        f"Non-retryable Bedrock streaming error: {error_code}",
                        exc_info=error,
                    )
                    raise

                if retry >= max_retries:
                    logger.error(
                        f"Bedrock streaming failed with {error_code} after "
                        f"{max_retries} retries"
                    )
                    raise

                # Calculate delay with exponential backoff and jitter
                wait_time = calculate_backoff_delay(
                    attempt=retry,
                    base_delay=RETRY_BASE_DELAY,
                    max_delay=RETRY_MAX_DELAY,
                    min_delay=RETRY_MIN_DELAY,
                    jitter_factor=RETRY_JITTER_FACTOR,
                )

                # Log and backoff
                logger.warning(
                    f"Bedrock streaming received {error_code}, "
                    f"backing off ({wait_time:.1f}s) and retrying "
                    f"(attempt {retry + 1}/{max_retries})"
                )

                await asyncio.sleep(wait_time)
                retry += 1

            except Exception as e:
                # Unexpected error (not ClientError) - propagate immediately
                logger.error(f"Unexpected Bedrock streaming error: {e}", exc_info=e)
                raise

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
        disable_safety_filters: bool = False,
    ) -> Union[AssistantMessage, "ToolCallMessage"]:
        if disable_safety_filters:
            logger.debug(
                "[BEDROCK] disable_safety_filters has no effect — Bedrock guardrails "
                "are additive (they ADD filtering when configured) and there is no "
                "API parameter that weakens the underlying model's intrinsic safety."
            )
        logger.info("Request to AWS Bedrock made")
        logger_llm.debug(
            "\n---\n".join(
                [f"{message.__repr__(truncate=False)}" for message in messages]
            )
        )
        logger_llm.info(
            f"Model params: {self.model_id}, temp: {self.temperature}, top_p: {self.top_p}, tools: {tools}"
        )

        converse_params = self._build_converse_params(
            messages, tools, structured_output
        )

        try:
            response = self._execute_with_retry(
                lambda: self.client.converse(**converse_params),
                "AWS Bedrock converse",
            )
            logger.info("AWS Bedrock response received")
        except (ClientError, Exception) as e:
            logger.error(
                f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}", exc_info=e
            )
            raise

        return self._process_converse_response(
            response["output"],
            structured_output,
            stop_reason=response.get("stopReason"),
        )

    @staticmethod
    def get_name():
        return "Bedrock"

    @staticmethod
    def get_available_models(
        region: str = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        timeout: int = BEDROCK_DEFAULT_TIMEOUT,
    ) -> List[str]:
        client = get_bedrock_client_by_region(
            client_type="bedrock", region=region, timeout=timeout
        )

        return [
            model["modelName"]
            for model in client.list_foundation_models(byOutputModality="TEXT")[
                "modelSummaries"
            ]
        ]

    def count_tokens(self, content: Union[str, Message, List[Message]]) -> int:
        """
        Count tokens for Bedrock models.

        Different model families use different counting methods:
        - Anthropic Claude: Uses Bedrock's CountTokens API (batch support)
        - Meta Llama: Uses HuggingFace transformers (optional dependency)
        - Mistral: Uses HuggingFace transformers (optional dependency)
        - Amazon Titan: Not supported (proprietary tokenizer)

        Args:
            content: A string, single Message, or list of Messages

        Returns:
            Number of tokens

        Raises:
            TokenCountingError: If counting cannot be performed
        """
        model_lower = self.model_id.lower()

        if "anthropic" in model_lower or "claude" in model_lower:
            return self._count_tokens_bedrock_claude(content)
        elif "llama" in model_lower or "meta" in model_lower:
            return self._count_tokens_llama(content)
        elif "mistral" in model_lower:
            return self._count_tokens_mistral(content)
        else:
            raise TokenCountingError(
                f"Token counting for {self.model_id} is not supported. "
                f"Supported model families: Anthropic Claude (API), Meta Llama (transformers), "
                f"Mistral (transformers). Amazon Titan uses a proprietary tokenizer."
            )

    def _count_tokens_bedrock_claude(
        self, content: Union[str, Message, List[Message]]
    ) -> int:
        """
        Count tokens using Bedrock's CountTokens API for Claude models.

        Supports batch counting of multiple messages in a single API call.
        Falls back to local estimation for older Claude models (e.g., Claude 3)
        that don't support Bedrock's CountTokens.
        """
        if isinstance(content, str):
            messages = [{"role": "user", "content": [{"text": content}]}]
        elif isinstance(content, list):
            messages = self._convert_messages_for_token_counting(content)
        else:
            messages = self._convert_messages_for_token_counting([content])

        converse_input = {"messages": messages}

        try:
            response = self.client.count_tokens(
                modelId=self.model_id,
                input={"converse": converse_input},
            )
            return response["inputTokens"]
        except ClientError as e:
            # Fall back to local estimation for older Claude models (e.g., Claude 3)
            # that don't support Bedrock's CountTokens API
            error_message = str(e)
            if "doesn't support counting tokens" in error_message:
                return self._estimate_tokens_locally(content)
            raise TokenCountingError(f"Bedrock API error: {e}")
        except Exception as e:
            raise TokenCountingError(f"Failed to count tokens: {e}")

    def _estimate_tokens_locally(
        self, content: Union[str, Message, List[Message]]
    ) -> int:
        """
        Estimate tokens locally using character count heuristic with length scaling.

        Used as fallback for Claude models that don't support Bedrock's CountTokens
        API (e.g., Claude 3 models in regions like ca-central-1).

        Uses a scaled approach based on observed tokenization patterns:
        - Short text (<500 chars): ~3.5 chars/token, 20% margin
        - Medium text (500-2000 chars): ~4.0 chars/token, 15% margin
        - Long text (>2000 chars): ~4.5 chars/token, 10% margin

        This keeps estimates conservative (always overestimates) while avoiding
        excessive waste on longer documents where the context window is precious.
        """
        if isinstance(content, str):
            char_count = len(content)
        elif isinstance(content, list):
            char_count = sum(self._count_message_chars(m) for m in content)
        else:
            char_count = self._count_message_chars(content)

        # Scale chars_per_token and margin based on text length
        # Based on testing with LICENSE file (~4.0-4.3 chars/tok) and formal prose (~6+ chars/tok)
        if char_count < 500:
            chars_per_token = 3.5
            margin = 1.20  # 20% for short text where overhead dominates
        elif char_count < 2000:
            chars_per_token = 3.8
            margin = 1.15  # 15% for medium text
        else:
            chars_per_token = 4.0
            margin = 1.15  # 15% for long text (legal/formal docs ~4.0-4.3 chars/tok)

        base_estimate = char_count / chars_per_token
        return int(base_estimate * margin + 0.5)

    def _count_message_chars(self, message: Message) -> int:
        """Count characters in a message for token estimation."""
        text = message.get_content_as_string()
        char_count = len(text)

        # Add overhead for message structure (~20 chars for role, delimiters)
        char_count += 20

        # For tool calls, include the JSON arguments
        if isinstance(message, ToolCallMessage):
            for tc in message.tool_calls:
                char_count += len(tc.name) + len(tc.arguments) + len(tc.id)

        # For tool results, include metadata
        if isinstance(message, ToolResultMessage):
            if message.call_id:
                char_count += len(message.call_id)
            if message.function_name:
                char_count += len(message.function_name)

        return char_count

    def _get_llama_tokenizer(self, tokenizer_name: str):
        """
        Get cached Llama tokenizer, loading it if needed.

        Uses class-level cache to avoid expensive HuggingFace tokenizer loading
        on every count_tokens() call.
        """
        if tokenizer_name not in BedrockModel._llama_tokenizer_cache:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise TokenCountingError(
                    "Token counting for Llama requires 'transformers'. "
                    "Install with: pip install transformers sentencepiece"
                )

            try:
                BedrockModel._llama_tokenizer_cache[tokenizer_name] = (
                    AutoTokenizer.from_pretrained(tokenizer_name)
                )
            except Exception as e:
                raise TokenCountingError(
                    f"Failed to load Llama tokenizer '{tokenizer_name}': {e}. "
                    f"You may need to authenticate with HuggingFace: huggingface-cli login"
                )

        return BedrockModel._llama_tokenizer_cache[tokenizer_name]

    def _count_tokens_llama(self, content: Union[str, Message, List[Message]]) -> int:
        """
        Count tokens using HuggingFace tokenizer for Llama models.

        Requires optional transformers dependency. Tokenizers are cached at
        class level for performance.
        """
        # Determine tokenizer based on model version
        tokenizer_name = "meta-llama/Meta-Llama-3-8B"
        if "llama-2" in self.model_id.lower():
            tokenizer_name = "meta-llama/Llama-2-7b-hf"

        tokenizer = self._get_llama_tokenizer(tokenizer_name)

        if isinstance(content, str):
            return len(tokenizer.encode(content))
        elif isinstance(content, list):
            total = 0
            for message in content:
                total += self._count_single_message_llama(message, tokenizer)
            return total
        else:
            return self._count_single_message_llama(content, tokenizer)

    def _count_single_message_llama(self, message: Message, tokenizer) -> int:
        """Count tokens for a single message using Llama tokenizer."""
        text = message.get_content_as_string()

        # Warn about multimodal content
        if isinstance(message.content, list):
            for chunk in message.content:
                if isinstance(chunk, MultimodalChunk):
                    import warnings

                    warnings.warn(
                        "Llama token counting does not include multimodal content. "
                        "Only text tokens are counted.",
                        stacklevel=3,
                    )
                    break

        # Add message overhead (~4 tokens)
        return len(tokenizer.encode(text)) + 4

    def _get_mistral_tokenizer(self):
        """
        Get cached Mistral tokenizer, loading it if needed.

        Uses class-level cache to avoid expensive HuggingFace tokenizer loading
        on every count_tokens() call.
        """
        if BedrockModel._mistral_tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise TokenCountingError(
                    "Token counting for Mistral requires 'transformers'. "
                    "Install with: pip install transformers sentencepiece"
                )

            tokenizer_name = "mistralai/Mistral-7B-v0.1"
            try:
                BedrockModel._mistral_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name
                )
            except Exception as e:
                raise TokenCountingError(
                    f"Failed to load Mistral tokenizer '{tokenizer_name}': {e}."
                )

        return BedrockModel._mistral_tokenizer

    def _count_tokens_mistral(self, content: Union[str, Message, List[Message]]) -> int:
        """
        Count tokens using HuggingFace tokenizer for Mistral models.

        Requires optional transformers dependency. Tokenizer is cached at
        class level for performance.
        """
        tokenizer = self._get_mistral_tokenizer()

        if isinstance(content, str):
            return len(tokenizer.encode(content))
        elif isinstance(content, list):
            total = 0
            for message in content:
                total += self._count_single_message_mistral(message, tokenizer)
            return total
        else:
            return self._count_single_message_mistral(content, tokenizer)

    def _count_single_message_mistral(self, message: Message, tokenizer) -> int:
        """Count tokens for a single message using Mistral tokenizer."""
        text = message.get_content_as_string()
        return len(tokenizer.encode(text)) + 4  # Add message overhead

    def _convert_messages_for_token_counting(
        self, messages: List[Message]
    ) -> List[dict]:
        """
        Convert messages to Bedrock format for token counting.

        Filters out system messages and converts the rest.
        """
        non_system_messages = [
            m
            for m in messages
            if m.role in [ROLE_USER, ROLE_ASSISTANT, "tool_call", "tool_result"]
        ]
        return self._convert_messages_for_bedrock(non_system_messages)

    async def stream_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
        disable_safety_filters: bool = False,
    ) -> AsyncIterator["StreamChunk"]:
        """
        Stream the assistant message response from AWS Bedrock.

        Yields StreamChunk objects for each streaming event.

        Implements retry logic for pre-stream errors (ThrottlingException,
        ServiceUnavailableException). Mid-stream errors cannot be retried
        and will propagate to caller.

        Since boto3 doesn't have native async support, this method runs the
        synchronous converse_stream call in a thread pool executor and yields
        chunks asynchronously.
        """
        if disable_safety_filters:
            logger.debug(
                "[BEDROCK] disable_safety_filters has no effect — Bedrock guardrails "
                "are additive and there is no API parameter that weakens intrinsic safety."
            )
        logger.info("Request to AWS Bedrock (streaming) made")
        logger_llm.info(
            f"Model params: {self.model_id}, temp: {self.temperature}, top_p: {self.top_p}, tools: {tools}"
        )

        converse_params = self._build_converse_params(
            messages, tools, structured_output
        )

        # Get region from existing client for creating streaming client
        region_name = self.client.meta.region_name

        # Create a fresh client for this streaming operation
        # This avoids thread-safety issues with sharing clients
        sdk_max_attempts = 1 if self.retry_config is not None else None
        streaming_client = create_bedrock_client_for_streaming(
            region=region_name,
            timeout=self.timeout,
            sdk_max_attempts=sdk_max_attempts,
        )

        if self.retry_config is not None:

            async def acquire_stream():
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: streaming_client.converse_stream(**converse_params),
                )
                stream = response.get("stream")
                if not stream:
                    raise ValueError("No stream returned from converse_stream")
                return self._iterate_stream_events(stream)

            chunk_stream = stream_with_retry_config(
                self.retry_config, acquire_stream, "Bedrock streaming"
            )
        else:
            # Use retry wrapper for pre-stream errors (ThrottlingException, etc.)
            chunk_stream = self._stream_with_retry(
                streaming_client,
                converse_params,
                MAX_RETRIES,
            )

        async for chunk in chunk_stream:
            yield chunk

    def _extract_usage_from_event(self, event: dict) -> Optional[dict]:
        """Extract usage statistics from a metadata event."""
        if "metadata" not in event:
            return None
        metadata = event["metadata"]
        if "usage" not in metadata:
            return None
        return {
            "input_tokens": metadata["usage"].get("inputTokens", 0),
            "output_tokens": metadata["usage"].get("outputTokens", 0),
        }

    def _track_reasoning_content(
        self, event: dict, thinking_block_state: Dict[int, Dict[str, str]]
    ) -> None:
        """Track reasoning content deltas for building complete thinking blocks."""
        if "contentBlockDelta" not in event:
            return

        block_delta = event["contentBlockDelta"]
        index = block_delta.get("contentBlockIndex", 0)
        delta = block_delta.get("delta", {})

        if "reasoningContent" not in delta:
            return

        reasoning = delta["reasoningContent"]
        if index not in thinking_block_state:
            thinking_block_state[index] = {"text": "", "signature": ""}

        if "text" in reasoning:
            thinking_block_state[index]["text"] += reasoning["text"]
        if "signature" in reasoning:
            thinking_block_state[index]["signature"] = reasoning["signature"]

    def _build_thinking_blocks(
        self, thinking_block_state: Dict[int, Dict[str, str]]
    ) -> List[dict]:
        """Build complete thinking blocks from accumulated state."""
        thinking_blocks = []
        for index in sorted(thinking_block_state.keys()):
            block = thinking_block_state[index]
            if block["text"] or block["signature"]:
                thinking_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block["text"],
                        "signature": block["signature"],
                    }
                )
        return thinking_blocks

    async def _iterate_stream_events(self, stream) -> AsyncIterator["StreamChunk"]:
        """
        Iterate over boto3 EventStream and yield StreamChunks.

        Since EventStream iteration is synchronous, we wrap each iteration
        in run_in_executor to avoid blocking the event loop.

        Detects mid-stream throttling errors and logs them with actionable
        guidance, as they cannot be automatically retried.
        """
        loop = asyncio.get_running_loop()
        iterator = iter(stream)
        usage = None

        # Track thinking blocks with signatures for MESSAGE_END
        # Key: contentBlockIndex, Value: {"text": accumulated_text, "signature": signature}
        thinking_block_state: Dict[int, Dict[str, str]] = {}

        # Sentinel to detect end of iteration - StopIteration can't propagate through run_in_executor
        _STREAM_END = object()

        def _next_event():
            try:
                return next(iterator)
            except StopIteration:
                return _STREAM_END

        while True:
            try:
                event = await asyncio.wait_for(
                    loop.run_in_executor(None, _next_event),
                    timeout=BEDROCK_STREAM_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                raise StreamingError(
                    f"Bedrock stream timed out waiting for next event "
                    f"(timeout: {BEDROCK_STREAM_TIMEOUT_SECONDS}s)"
                )
            except Exception as e:
                # Detect mid-stream throttling errors (EventStreamError with throttling)
                # AWS returns these in camelCase format which boto3 can't properly unmarshal
                error_str = str(e).lower()
                if "throttling" in error_str or "too many requests" in error_str:
                    logger.error(
                        f"Mid-stream throttling detected in Bedrock EventStream. "
                        f"This error cannot be automatically retried. "
                        f"Consider: (1) Increasing boto3 Config retries, "
                        f"(2) Implementing application-level rate limiting, "
                        f"(3) Using Provisioned Throughput. Error: {e}"
                    )
                raise

            if event is _STREAM_END:
                break

            # Handle metadata events (usage stats)
            event_usage = self._extract_usage_from_event(event)
            if event_usage:
                usage = event_usage
                continue

            # Track reasoning content for thinking blocks
            self._track_reasoning_content(event, thinking_block_state)

            # Convert and yield chunk
            chunk = self._convert_stream_event_to_chunk(event)
            if chunk is not None:
                yield chunk

        # Build complete thinking blocks with signatures
        thinking_blocks = self._build_thinking_blocks(thinking_block_state)

        # Yield MESSAGE_END with usage and thinking blocks
        yield StreamChunk(
            event_type=StreamEventType.MESSAGE_END,
            usage=usage or {},
            thinking_blocks=thinking_blocks if thinking_blocks else None,
        )

    def _convert_stream_event_to_chunk(self, event: dict) -> Optional["StreamChunk"]:
        """
        Convert a Bedrock streaming event to a StreamChunk.

        Returns None for events we don't need to expose.
        """
        # messageStart - start of message
        if "messageStart" in event:
            return None  # We don't expose message start

        # contentBlockStart - start of a content block
        if "contentBlockStart" in event:
            block_start = event["contentBlockStart"]
            index = block_start.get("contentBlockIndex", 0)
            start = block_start.get("start", {})

            # Tool use start
            if "toolUse" in start:
                tool_use = start["toolUse"]
                return StreamChunk(
                    event_type=StreamEventType.TOOL_USE_START,
                    index=index,
                    tool_call_id=tool_use.get("toolUseId"),
                    tool_name=tool_use.get("name"),
                )
            # Text content start
            elif "text" in start:
                return StreamChunk(
                    event_type=StreamEventType.CONTENT_BLOCK_START,
                    index=index,
                    block_type="text",
                )

            return None

        # contentBlockDelta - incremental content
        if "contentBlockDelta" in event:
            block_delta = event["contentBlockDelta"]
            index = block_delta.get("contentBlockIndex", 0)
            delta = block_delta.get("delta", {})

            # Text delta
            if "text" in delta:
                return StreamChunk(
                    event_type=StreamEventType.TEXT_DELTA,
                    text=delta["text"],
                    index=index,
                )
            # Tool use input delta
            elif "toolUse" in delta:
                tool_delta = delta["toolUse"]
                return StreamChunk(
                    event_type=StreamEventType.TOOL_USE_DELTA,
                    tool_arguments_delta=tool_delta.get("input", ""),
                    index=index,
                )
            # Reasoning/thinking content delta (for Claude models with extended thinking)
            elif "reasoningContent" in delta:
                reasoning_content = delta["reasoningContent"]
                if "text" in reasoning_content:
                    return StreamChunk(
                        event_type=StreamEventType.THINKING_DELTA,
                        text=reasoning_content["text"],
                        index=index,
                    )

            return None

        # contentBlockStop - end of a content block
        if "contentBlockStop" in event:
            block_stop = event["contentBlockStop"]
            return StreamChunk(
                event_type=StreamEventType.CONTENT_BLOCK_STOP,
                index=block_stop.get("contentBlockIndex", 0),
            )

        # messageStop - end of message (stop reason but not final)
        if "messageStop" in event:
            return StreamChunk(
                event_type=StreamEventType.MESSAGE_DELTA,
            )

        return None

    def __deepcopy__(self, memo_dict):
        return BedrockModel(
            model_id=self.model_id,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            region_name=self.client.meta.region_name,
            thinking_config=self.thinking_config,
            timeout=self.timeout,
            retry_config=self.retry_config,
        )
