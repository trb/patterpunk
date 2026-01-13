import json
import time
from abc import ABC
from dataclasses import dataclass
from typing import (
    AsyncIterator,
    List,
    Optional,
    Callable,
    get_args,
    Set,
    Union,
    Literal,
)

from patterpunk.config.providers.anthropic import (
    anthropic,
    anthropic_async,
    ANTHROPIC_DEFAULT_TEMPERATURE,
    ANTHROPIC_DEFAULT_TOP_P,
    ANTHROPIC_DEFAULT_TOP_K,
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    ANTHROPIC_DEFAULT_TIMEOUT,
)
from patterpunk.config.defaults import MAX_RETRIES
from patterpunk.llm.messages.base import Message
from patterpunk.llm.messages.roles import ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.models.base import Model, TokenCountingError
from patterpunk.llm.thinking import ThinkingConfig as UnifiedThinkingConfig
from patterpunk.llm.types import ToolDefinition, CacheChunk, ToolCall
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.chunks import MultimodalChunk, TextChunk
from patterpunk.llm.messages.cache import get_multimodal_chunks, has_multimodal_content
from patterpunk.lib.structured_output import has_model_schema, get_model_schema
from patterpunk.logger import logger


if anthropic:
    from anthropic import APIError


@dataclass
class ThinkingConfig:
    type: Literal["enabled"] = "enabled"
    budget_tokens: int = 4000


class AnthropicRateLimitError(Exception):
    pass


class AnthropicMaxTokensError(Exception):
    pass


class AnthropicNotImplemented(Exception):
    pass


class AnthropicAPIError(Exception):
    pass


class AnthropicModel(Model, ABC):
    def __init__(
        self,
        model: str,
        temperature: float = ANTHROPIC_DEFAULT_TEMPERATURE,
        top_p: float = ANTHROPIC_DEFAULT_TOP_P,
        top_k: int = ANTHROPIC_DEFAULT_TOP_K,
        max_tokens: int = ANTHROPIC_DEFAULT_MAX_TOKENS,
        timeout: int = ANTHROPIC_DEFAULT_TIMEOUT,
        thinking_config: Optional[UnifiedThinkingConfig] = None,
    ):
        thinking = None
        if thinking_config is not None:
            if thinking_config.token_budget is not None:
                budget_tokens = min(thinking_config.token_budget, 128000)
            else:
                effort_to_tokens = {"low": 2000, "medium": 8000, "high": 24000}
                budget_tokens = effort_to_tokens[thinking_config.effort]
            thinking = ThinkingConfig(type="enabled", budget_tokens=budget_tokens)

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.thinking = thinking
        self.thinking_config = thinking_config

    def _convert_tools_to_anthropic_format(self, tools: ToolDefinition) -> List[dict]:
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                anthropic_tool = {
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"],
                }
                anthropic_tools.append(anthropic_tool)
        return anthropic_tools

    def _create_structured_output_tool(self, structured_output: object) -> dict:
        if not has_model_schema(structured_output):
            raise ValueError(
                "structured_output must be a Pydantic model with schema support"
            )

        schema = get_model_schema(structured_output)

        return {
            "name": "provide_structured_response",
            "description": "Provide the response in the exact structured format specified by the schema.",
            "input_schema": schema,
        }

    def _format_reasoning_to_structured_output(
        self,
        reasoning_content: str,
        structured_output: object,
        original_messages: List[Message],
        thinking_blocks: Optional[List[dict]] = None,
    ) -> AssistantMessage:
        """
        Reasoning models can't use tool_choice constraints, so we use a two-model approach:
        first the reasoning model generates analysis, then Haiku formats it to structured JSON.
        We use Haiku because it's fast, cheap, and reliable for formatting tasks.
        """
        logger.info(
            "[ANTHROPIC] Formatting reasoning output to structured JSON using Claude 3.5 Haiku"
        )

        structured_output_tool = self._create_structured_output_tool(structured_output)

        user_context = ""
        for msg in reversed(original_messages):
            if msg.role == ROLE_USER:
                user_context = msg.get_content_as_string()
                break

        formatting_prompt = f"""Based on the following reasoning and analysis, extract and format the information into the exact JSON structure specified by the tool schema.

Original user request: {user_context}

Reasoning and analysis from Claude:
{reasoning_content}

Please extract the relevant information from this reasoning and format it exactly according to the JSON schema provided in the tool. Do not add any additional text or explanation - just call the tool with the properly formatted data."""

        haiku_params = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": formatting_prompt}],
            "max_tokens": 4096,
            "temperature": 0.1,
            "tools": [structured_output_tool],
            "tool_choice": {"type": "tool", "name": "provide_structured_response"},
        }

        try:
            formatting_response = anthropic.messages.create(**haiku_params)

            for block in formatting_response.content:
                if (
                    block.type == "tool_use"
                    and block.name == "provide_structured_response"
                ):
                    if hasattr(block, "input") and block.input:
                        try:
                            parsed_output = structured_output.model_validate(
                                block.input
                            )
                            structured_response_content = json.dumps(
                                block.input, indent=2
                            )

                            logger.info(
                                "[ANTHROPIC] Successfully formatted reasoning output to structured JSON"
                            )
                            return AssistantMessage(
                                structured_response_content,
                                structured_output=structured_output,
                                parsed_output=parsed_output,
                                thinking_blocks=thinking_blocks,
                            )
                        except Exception as e:
                            logger.error(
                                f"[ANTHROPIC] Failed to parse structured output from Haiku formatting: {e}"
                            )

            logger.warning(
                "[ANTHROPIC] Haiku formatting failed, falling back to reasoning content"
            )
            return AssistantMessage(
                reasoning_content,
                structured_output=structured_output,
                thinking_blocks=thinking_blocks,
            )

        except Exception as e:
            logger.error(
                f"[ANTHROPIC] Error in two-model structured output approach: {e}"
            )
            return AssistantMessage(
                reasoning_content,
                structured_output=structured_output,
                thinking_blocks=thinking_blocks,
            )

    def _parse_model_version(self) -> tuple[int, int]:
        import re

        # Claude 3.x with minor version: claude-3-7-sonnet-20250219, claude-3-5-haiku-20241022
        claude3_minor_match = re.search(
            r"claude-3-(\d+)-(?:opus|sonnet|haiku)", self.model
        )
        if claude3_minor_match:
            return (3, int(claude3_minor_match.group(1)))

        # Claude 3.0 base format: claude-3-haiku-20240307, claude-3-opus-20240229
        claude3_base_match = re.search(
            r"claude-3-(?:opus|sonnet|haiku)-\d{8}", self.model
        )
        if claude3_base_match:
            return (3, 0)

        # Claude 4+ format: claude-opus-4-20250514, claude-sonnet-4-5-20250614
        # Need to distinguish between minor version and date
        claude4plus_match = re.search(
            r"claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d+))?-(\d{8})", self.model
        )
        if claude4plus_match:
            major = int(claude4plus_match.group(1))
            # Group 2 is minor version, Group 3 is date (8 digits)
            minor_str = claude4plus_match.group(2)
            minor = int(minor_str) if minor_str else 0
            return (major, minor)

        return (0, 0)

    def _is_reasoning_model(self) -> bool:
        major, minor = self._parse_model_version()

        if major >= 4:
            return True
        elif major == 3 and minor >= 7:
            return True

        return False

    def _get_compatible_params(self, api_params: dict) -> dict:
        major, minor = self._parse_model_version()

        # Claude 4+ models with thinking mode: remove top_p/top_k, force temperature=1.0
        if major >= 4 and self.thinking:
            compatible_params = api_params.copy()
            if "top_p" in compatible_params:
                del compatible_params["top_p"]
            if "top_k" in compatible_params:
                del compatible_params["top_k"]
            compatible_params["temperature"] = 1.0
            return compatible_params

        # Claude 4+ models without thinking: temperature and top_p cannot both be specified
        if major >= 4:
            has_temp = "temperature" in api_params
            has_top_p = "top_p" in api_params

            if has_temp and has_top_p:
                # If top_p is at default (1.0), omit it (effectively a no-op anyway)
                # This handles the common case where defaults are used
                if api_params.get("top_p") == 1.0:
                    compatible_params = api_params.copy()
                    del compatible_params["top_p"]
                    if "top_k" in compatible_params:
                        del compatible_params["top_k"]
                    return compatible_params

                # If both are non-default, user explicitly wants both - raise error
                if (
                    api_params.get("temperature") != 0.7
                    or api_params.get("top_p") != 1.0
                ):
                    raise ValueError(
                        f"Claude 4+ models do not support both 'temperature' and 'top_p' simultaneously. "
                        f"Please use only one. Current values: temperature={api_params.get('temperature')}, "
                        f"top_p={api_params.get('top_p')}. "
                        f"Anthropic recommends using 'temperature' for most use cases."
                    )

        # For Claude 3.x with thinking mode (existing behavior)
        if self.thinking:
            compatible_params = api_params.copy()
            if "top_p" in compatible_params:
                del compatible_params["top_p"]
            if "top_k" in compatible_params:
                del compatible_params["top_k"]
            compatible_params["temperature"] = 1.0
            return compatible_params

        return api_params

    def _prepare_system_prompt(
        self, messages: List[Message]
    ) -> Optional[Union[str, List[dict]]]:
        system_content = self._convert_system_messages_for_anthropic(messages)
        system_prompt = None
        if system_content:
            if len(system_content) == 1 and system_content[0].get("type") == "text":
                if "cache_control" not in system_content[0]:
                    system_prompt = system_content[0]["text"]
                else:
                    system_prompt = system_content
            else:
                system_prompt = system_content
        return system_prompt

    def _build_base_api_parameters(
        self, messages: List[Message], system_prompt: Optional[Union[str, List[dict]]]
    ) -> dict:
        api_params = {
            "model": self.model,
            "messages": self._convert_messages_for_anthropic(
                [
                    message
                    for message in messages
                    if message.role
                    in [ROLE_USER, ROLE_ASSISTANT, "tool_call", "tool_result"]
                ]
            ),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "timeout": self.timeout,
        }

        if system_prompt is not None:
            api_params["system"] = system_prompt

        return api_params

    def _apply_thinking_configuration(self, api_params: dict) -> dict:
        if self.thinking and self._is_reasoning_model():
            api_params["thinking"] = {
                "type": self.thinking.type,
                "budget_tokens": self.thinking.budget_tokens,
            }
            # Enable interleaved thinking for tool calls with Claude 4.5+ models
            # This allows thinking blocks between tool calls for more sophisticated reasoning
            major, minor = self._parse_model_version()
            if major >= 4 and minor >= 5:
                api_params["extra_headers"] = {
                    "anthropic-beta": "interleaved-thinking-2025-05-14"
                }
        # Always apply parameter compatibility checks for Claude 4+ models
        # to handle temperature/top_p conflicts and thinking mode requirements
        api_params = self._get_compatible_params(api_params)
        return api_params

    def _initialize_retry_state(self) -> tuple[int, int]:
        retry_count = 0
        wait_time = 60
        return retry_count, wait_time

    def _handle_api_error(
        self, error: "APIError", retry_count: int, wait_time: int
    ) -> tuple[bool, int, int]:
        if (
            getattr(error, "status_code", None) == 429
            or "rate_limit_error" in str(error).lower()
        ):
            if retry_count >= MAX_RETRIES:
                raise AnthropicRateLimitError(
                    f"Rate limit exceeded after {retry_count} retries"
                ) from error

            logger.warning(
                f"Rate limit hit, attempt {retry_count + 1}/{MAX_RETRIES}. "
                f"Waiting {wait_time} seconds before retry."
            )

            time.sleep(wait_time)
            new_retry_count = retry_count + 1
            new_wait_time = int(wait_time * 1.5)
            return True, new_retry_count, new_wait_time
        else:
            raise error

    def _configure_regular_tools(
        self, api_params: dict, tools: Optional[ToolDefinition]
    ) -> dict:
        if tools:
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
            if anthropic_tools:
                api_params["tools"] = anthropic_tools
        return api_params

    def _configure_structured_output_non_reasoning(
        self,
        api_params: dict,
        tools: Optional[ToolDefinition],
        structured_output: object,
    ) -> dict:
        structured_output_tool = self._create_structured_output_tool(structured_output)

        anthropic_tools = []
        if tools:
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)

        anthropic_tools.append(structured_output_tool)
        api_params["tools"] = anthropic_tools

        api_params["tool_choice"] = {
            "type": "tool",
            "name": "provide_structured_response",
        }
        return api_params

    def _configure_reasoning_structured_output_auto(
        self,
        api_params: dict,
        tools: Optional[ToolDefinition],
        structured_output: object,
    ) -> dict:
        structured_output_tool = self._create_structured_output_tool(structured_output)

        anthropic_tools = []
        if tools:
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)

        anthropic_tools.append(structured_output_tool)
        api_params["tools"] = anthropic_tools

        api_params["tool_choice"] = {"type": "auto"}
        return api_params

    def _configure_reasoning_tools_fallback(
        self, api_params: dict, tools: Optional[ToolDefinition]
    ) -> dict:
        if tools:
            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
            api_params["tools"] = anthropic_tools
            api_params["tool_choice"] = {"type": "auto"}
        return api_params

    def _configure_tools_and_structured_output(
        self,
        api_params: dict,
        tools: Optional[ToolDefinition],
        structured_output: Optional[object],
    ) -> dict:
        if structured_output and has_model_schema(structured_output):
            if self.thinking and self._is_reasoning_model():
                return self._configure_reasoning_structured_output_auto(
                    api_params, tools, structured_output
                )
            else:
                return self._configure_structured_output_non_reasoning(
                    api_params, tools, structured_output
                )
        else:
            return self._configure_regular_tools(api_params, tools)

    def _extract_thinking_blocks(self, response) -> List[dict]:
        """
        Extract thinking blocks from Anthropic API response.
        Returns list of thinking block dictionaries in their original format.
        """
        thinking_blocks = []
        for block in response.content:
            if block.type == "thinking":
                thinking_block = {
                    "type": "thinking",
                    "thinking": block.thinking,
                }
                # Include signature if present
                if hasattr(block, "signature") and block.signature:
                    thinking_block["signature"] = block.signature
                thinking_blocks.append(thinking_block)
            elif block.type == "redacted_thinking":
                redacted_block = {
                    "type": "redacted_thinking",
                    "data": block.data,
                }
                thinking_blocks.append(redacted_block)
        return thinking_blocks

    def _extract_reasoning_content(self, response) -> str:
        reasoning_parts = []
        for block in response.content:
            if block.type == "text":
                reasoning_parts.append(block.text)
            elif (
                block.type == "tool_use" and block.name != "provide_structured_response"
            ):
                reasoning_parts.append(
                    f"Tool call: {block.name} with arguments: {json.dumps(block.input)}"
                )
        return "\n".join(reasoning_parts)

    def _parse_structured_output_from_tool_call(
        self,
        block,
        structured_output: object,
        thinking_blocks: Optional[List[dict]] = None,
    ) -> Optional[AssistantMessage]:
        if block.name == "provide_structured_response" and structured_output:
            if hasattr(block, "input") and block.input:
                try:
                    parsed_output = structured_output.model_validate(block.input)
                    structured_response_content = json.dumps(block.input, indent=2)

                    return AssistantMessage(
                        structured_response_content,
                        structured_output=structured_output,
                        parsed_output=parsed_output,
                        thinking_blocks=thinking_blocks,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to parse structured output from tool call: {e}"
                    )
        return None

    def _convert_tool_call_block(self, block) -> ToolCall:
        arguments = "{}"
        if hasattr(block, "input") and block.input:
            try:
                arguments = json.dumps(block.input)
            except (TypeError, ValueError):
                arguments = str(block.input)

        return ToolCall(
            id=block.id,
            name=block.name,
            arguments=arguments,
        )

    def _assemble_text_content(
        self, response, structured_output: Optional[object]
    ) -> AssistantMessage:
        content = "\n".join(
            [block.text for block in response.content if block.type == "text"]
        )
        thinking_blocks = self._extract_thinking_blocks(response)
        return AssistantMessage(
            content,
            structured_output=structured_output,
            thinking_blocks=thinking_blocks,
        )

    def _validate_stop_reason(self, response) -> None:
        if response.stop_reason == "max_tokens":
            logger.warning("Anthropic response was cut off as the model hit MAX_TOKENS")
        elif response.stop_reason == "refusal":
            logger.warning(
                "Anthropic model refused to generate content for safety reasons"
            )

    def _execute_with_retry_loop(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition],
        structured_output: Optional[object],
    ) -> Union[Message, "ToolCallMessage"]:
        system_prompt = self._prepare_system_prompt(messages)
        retry_count, wait_time = self._initialize_retry_state()

        while True:
            try:
                api_params = self._build_base_api_parameters(messages, system_prompt)
                api_params = self._apply_thinking_configuration(api_params)

                if (
                    structured_output
                    and has_model_schema(structured_output)
                    and self.thinking
                    and self._is_reasoning_model()
                ):
                    logger.info(
                        f"[ANTHROPIC] Attempting reasoning model with auto tool choice for structured output"
                    )

                    api_params = self._configure_reasoning_structured_output_auto(
                        api_params, tools, structured_output
                    )

                    try:
                        reasoning_response = anthropic.messages.create(**api_params)
                        thinking_blocks = self._extract_thinking_blocks(
                            reasoning_response
                        )

                        for block in reasoning_response.content:
                            if (
                                block.type == "tool_use"
                                and block.name == "provide_structured_response"
                            ):
                                if hasattr(block, "input") and block.input:
                                    try:
                                        parsed_output = (
                                            structured_output.model_validate(
                                                block.input
                                            )
                                        )
                                        structured_response_content = json.dumps(
                                            block.input, indent=2
                                        )

                                        logger.info(
                                            "[ANTHROPIC] Successfully got structured output from reasoning model with auto tool choice"
                                        )
                                        return AssistantMessage(
                                            structured_response_content,
                                            structured_output=structured_output,
                                            parsed_output=parsed_output,
                                            thinking_blocks=thinking_blocks,
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"[ANTHROPIC] Failed to parse structured output from reasoning model: {e}"
                                        )
                                        break

                        logger.info(
                            "[ANTHROPIC] Reasoning model didn't use structured output tool, falling back to two-model approach"
                        )

                        reasoning_content = self._extract_reasoning_content(
                            reasoning_response
                        )
                        return self._format_reasoning_to_structured_output(
                            reasoning_content,
                            structured_output,
                            messages,
                            thinking_blocks,
                        )

                    except Exception as e:
                        logger.warning(
                            f"[ANTHROPIC] Error with reasoning model auto tool choice: {e}, falling back to two-model approach"
                        )

                        api_params = self._configure_reasoning_tools_fallback(
                            api_params, tools
                        )
                        reasoning_response = anthropic.messages.create(**api_params)
                        thinking_blocks = self._extract_thinking_blocks(
                            reasoning_response
                        )
                        reasoning_content = self._extract_reasoning_content(
                            reasoning_response
                        )
                        return self._format_reasoning_to_structured_output(
                            reasoning_content,
                            structured_output,
                            messages,
                            thinking_blocks,
                        )

                else:
                    api_params = self._configure_tools_and_structured_output(
                        api_params, tools, structured_output
                    )

                response = anthropic.messages.create(**api_params)

                if response.stop_reason in [
                    "end_turn",
                    "stop_sequence",
                    "max_tokens",
                    "refusal",
                ]:
                    self._validate_stop_reason(response)
                    return self._assemble_text_content(response, structured_output)
                elif response.stop_reason == "tool_use":
                    tool_calls = []
                    thinking_blocks = self._extract_thinking_blocks(response)

                    for block in response.content:
                        if block.type == "tool_use":
                            structured_result = (
                                self._parse_structured_output_from_tool_call(
                                    block, structured_output, thinking_blocks
                                )
                            )
                            if structured_result:
                                return structured_result

                            tool_call = self._convert_tool_call_block(block)
                            tool_calls.append(tool_call)

                    if tool_calls:
                        return ToolCallMessage(
                            tool_calls, thinking_blocks=thinking_blocks
                        )
                    else:
                        raise AnthropicAPIError(
                            "Tool use stop reason but no tool use blocks found in response"
                        )
                else:
                    raise AnthropicAPIError(
                        f"Unknown stop reason: {response.stop_reason}"
                    )

            except APIError as e:
                should_retry, retry_count, wait_time = self._handle_api_error(
                    e, retry_count, wait_time
                )
                if should_retry:
                    continue
        raise AnthropicAPIError(
            f"Unexpected outcome - out of retries, but neither error raised or message returned"
        )

    def _convert_content_to_anthropic_format(self, content) -> List[dict]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        anthropic_content = []
        session = None

        for chunk in content:
            if isinstance(chunk, TextChunk):
                anthropic_content.append({"type": "text", "text": chunk.content})

            elif isinstance(chunk, CacheChunk):
                content_block = {"type": "text", "text": chunk.content}

                if chunk.cacheable:
                    cache_control = {"type": "ephemeral"}
                    if chunk.ttl:
                        cache_control["ttl"] = int(chunk.ttl.total_seconds())
                    content_block["cache_control"] = cache_control

                anthropic_content.append(content_block)

            elif isinstance(chunk, MultimodalChunk):
                if hasattr(chunk, "file_id"):
                    content_block = {
                        "type": "document",
                        "source": {"type": "file", "file_id": chunk.file_id},
                    }
                    anthropic_content.append(content_block)
                    continue

                if chunk.source_type == "url":
                    if session is None:
                        try:
                            import requests

                            session = requests.Session()
                        except ImportError:
                            raise ImportError(
                                "requests library required for URL support with Anthropic"
                            )

                    chunk = chunk.download(session)

                media_type = chunk.media_type or "application/octet-stream"

                if media_type.startswith("image/"):
                    content_block = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": chunk.to_base64(),
                        },
                    }
                    anthropic_content.append(content_block)
                elif media_type == "application/pdf":
                    content_block = {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": chunk.to_base64(),
                        },
                    }
                    anthropic_content.append(content_block)

        return anthropic_content

    def upload_file_to_anthropic(self, chunk: MultimodalChunk) -> str:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=anthropic.api_key)

            import tempfile
            import os

            with tempfile.NamedTemporaryFile(
                suffix=f".{chunk.media_type.split('/')[-1] if chunk.media_type else 'bin'}",
                delete=False,
            ) as tmp_file:
                tmp_file.write(chunk.to_bytes())
                tmp_file_path = tmp_file.name

            try:
                with open(tmp_file_path, "rb") as f:
                    file_response = client.files.create(file=f, purpose="vision")

                return file_response.id
            finally:
                os.unlink(tmp_file_path)

        except Exception as e:
            logger.error(f"Failed to upload file to Anthropic: {e}")
            raise

    def _convert_messages_for_anthropic(self, messages: List[Message]) -> List[dict]:
        anthropic_messages = []

        for message in messages:
            if message.role == "tool_call":
                # Serialize ToolCallMessage as assistant message with tool_use content blocks
                content_blocks = []

                # CRITICAL: Include thinking blocks first, before tool_use blocks
                # Anthropic requires thinking blocks to be preserved in multi-turn conversations
                if hasattr(message, "thinking_blocks") and message.thinking_blocks:
                    content_blocks.extend(message.thinking_blocks)

                for tool_call in message.tool_calls:
                    # Parse arguments from JSON string
                    try:
                        arguments = json.loads(tool_call.arguments)
                    except (json.JSONDecodeError, KeyError):
                        arguments = {}

                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": arguments,
                        }
                    )

                anthropic_messages.append(
                    {"role": "assistant", "content": content_blocks}
                )

            elif message.role == "tool_result":
                # Validate required field for Anthropic
                if not message.call_id:
                    raise ValueError(
                        "Anthropic requires call_id (as tool_use_id) in ToolResultMessage. "
                        "Ensure ToolResultMessage is created with call_id from the original ToolCallMessage."
                    )

                # Serialize as USER message with tool_result content block
                # Anthropic requires tool results to be sent as user role messages
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.call_id,
                                "content": message.content,
                                "is_error": message.is_error,
                            }
                        ],
                    }
                )

            else:
                # Handle regular messages (user, assistant)
                content = []

                # For assistant messages with thinking blocks, include them first
                if (
                    message.role == ROLE_ASSISTANT
                    and hasattr(message, "thinking_blocks")
                    and message.thinking_blocks
                ):
                    content.extend(message.thinking_blocks)

                # Then add the main content
                if isinstance(message.content, list):
                    content.extend(
                        self._convert_content_to_anthropic_format(message.content)
                    )
                else:
                    content.append({"type": "text", "text": message.content})

                anthropic_messages.append({"role": message.role, "content": content})

        return anthropic_messages

    def _convert_system_messages_for_anthropic(
        self, messages: List[Message]
    ) -> List[dict]:
        system_content = []

        for message in messages:
            if message.role == ROLE_SYSTEM:
                if isinstance(message.content, list):
                    system_content.extend(
                        self._convert_content_to_anthropic_format(message.content)
                    )
                else:
                    system_content.append({"type": "text", "text": message.content})

        return system_content

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> Union[Message, "ToolCallMessage"]:
        return self._execute_with_retry_loop(messages, tools, structured_output)

    def _prepare_streaming_parameters(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition],
        structured_output: Optional[object],
    ) -> dict:
        system_prompt = self._prepare_system_prompt(messages)
        api_params = self._build_base_api_parameters(messages, system_prompt)
        api_params = self._apply_thinking_configuration(api_params)
        api_params = self._configure_tools_and_structured_output(
            api_params, tools, structured_output
        )
        api_params["stream"] = True
        return api_params

    @staticmethod
    def get_available_models() -> List[str]:
        return [model.id for model in anthropic.models.list()]

    @staticmethod
    def get_name():
        return "Anthropic"

    def _extract_system_prompt(
        self, content: Union[str, Message, List[Message]]
    ) -> Optional[Union[str, List]]:
        """
        Extract system prompt from content for API call.

        Returns None if no system messages, a string for simple text system prompts,
        or a list for complex system prompts with multiple parts.
        """
        if not isinstance(content, list):
            return None

        system_content = self._convert_system_messages_for_anthropic(content)
        if not system_content:
            return None

        # Simple text system prompt - return as string
        is_single_text = (
            len(system_content) == 1 and system_content[0].get("type") == "text"
        )
        if is_single_text:
            return system_content[0]["text"]

        return system_content

    def _prepare_count_tokens_params(
        self, content: Union[str, Message, List[Message]]
    ) -> dict:
        """
        Prepare parameters for count_tokens API call.

        Converts content to messages format and extracts system prompt.
        """
        if isinstance(content, str):
            messages = [{"role": "user", "content": content}]
        elif isinstance(content, list):
            messages = self._convert_messages_for_token_counting(content)
        else:
            messages = self._convert_messages_for_token_counting([content])

        params = {"model": self.model, "messages": messages}

        system_prompt = self._extract_system_prompt(content)
        if system_prompt:
            params["system"] = system_prompt

        return params

    def count_tokens(self, content: Union[str, Message, List[Message]]) -> int:
        """
        Count tokens using Anthropic's API.

        This makes an API call to Anthropic's count_tokens endpoint, which accurately
        counts all content types including images and PDFs. For batch counting of
        multiple messages, a single API call is made.

        Args:
            content: A string, single Message, or list of Messages

        Returns:
            Number of tokens

        Raises:
            TokenCountingError: If counting fails
        """
        try:
            params = self._prepare_count_tokens_params(content)
            response = anthropic.messages.count_tokens(**params)
            return response.input_tokens
        except APIError as e:
            raise TokenCountingError(f"Anthropic API error: {e}")
        except Exception as e:
            raise TokenCountingError(f"Failed to count tokens: {e}")

    async def count_tokens_async(
        self, content: Union[str, Message, List[Message]]
    ) -> int:
        """
        Count tokens using Anthropic's async API.

        Native async version using Anthropic's async client for better concurrency.

        Args:
            content: A string, single Message, or list of Messages

        Returns:
            Number of tokens

        Raises:
            TokenCountingError: If counting fails
        """
        try:
            params = self._prepare_count_tokens_params(content)
            response = await anthropic_async.messages.count_tokens(**params)
            return response.input_tokens
        except APIError as e:
            raise TokenCountingError(f"Anthropic API error: {e}")
        except Exception as e:
            raise TokenCountingError(f"Failed to count tokens: {e}")

    def _convert_messages_for_token_counting(
        self, messages: List[Message]
    ) -> List[dict]:
        """
        Convert messages to Anthropic format for token counting.

        Filters out system messages (handled separately) and converts the rest.
        """
        non_system_messages = [
            m
            for m in messages
            if m.role in [ROLE_USER, ROLE_ASSISTANT, "tool_call", "tool_result"]
        ]
        return self._convert_messages_for_anthropic(non_system_messages)

    async def stream_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
    ) -> AsyncIterator["StreamChunk"]:
        """
        Stream the assistant message response from Anthropic.

        Yields StreamChunk objects for each streaming event.
        """
        from patterpunk.llm.streaming import StreamChunk, StreamEventType

        # Build API parameters (same as sync, but we'll use async client)
        system_prompt = self._prepare_system_prompt(messages)
        api_params = self._build_base_api_parameters(messages, system_prompt)
        api_params = self._apply_thinking_configuration(api_params)
        api_params = self._configure_tools_and_structured_output(
            api_params, tools, structured_output
        )

        # Remove timeout from params (context manager handles it)
        api_params.pop("timeout", None)

        async with anthropic_async.messages.stream(**api_params) as stream:
            async for event in stream:
                chunk = self._convert_stream_event_to_chunk(event)
                if chunk is not None:
                    yield chunk

            # Get final message for complete data
            final_message = stream.current_message_snapshot

            # Extract thinking blocks with signatures from final message
            thinking_blocks = self._extract_thinking_blocks(final_message)

            # Yield final message end event
            yield StreamChunk(
                event_type=StreamEventType.MESSAGE_END,
                usage={
                    "input_tokens": final_message.usage.input_tokens,
                    "output_tokens": final_message.usage.output_tokens,
                },
                thinking_blocks=thinking_blocks if thinking_blocks else None,
            )

    def _convert_stream_event_to_chunk(self, event) -> Optional["StreamChunk"]:
        """
        Convert an Anthropic streaming event to a StreamChunk.

        Returns None for events we don't need to expose.
        """
        from patterpunk.llm.streaming import StreamChunk, StreamEventType

        event_type = getattr(event, "type", None)

        if event_type == "content_block_start":
            block = event.content_block
            if block.type == "thinking":
                return StreamChunk(
                    event_type=StreamEventType.CONTENT_BLOCK_START,
                    index=event.index,
                    block_type="thinking",
                )
            elif block.type == "text":
                return StreamChunk(
                    event_type=StreamEventType.CONTENT_BLOCK_START,
                    index=event.index,
                    block_type="text",
                )
            elif block.type == "tool_use":
                return StreamChunk(
                    event_type=StreamEventType.TOOL_USE_START,
                    index=event.index,
                    tool_call_id=block.id,
                    tool_name=block.name,
                )

        elif event_type == "content_block_delta":
            delta = event.delta
            delta_type = getattr(delta, "type", None)

            if delta_type == "thinking_delta":
                return StreamChunk(
                    event_type=StreamEventType.THINKING_DELTA,
                    text=delta.thinking,
                    index=event.index,
                )
            elif delta_type == "text_delta":
                return StreamChunk(
                    event_type=StreamEventType.TEXT_DELTA,
                    text=delta.text,
                    index=event.index,
                )
            elif delta_type == "input_json_delta":
                return StreamChunk(
                    event_type=StreamEventType.TOOL_USE_DELTA,
                    tool_arguments_delta=delta.partial_json,
                    index=event.index,
                )

        elif event_type == "content_block_stop":
            return StreamChunk(
                event_type=StreamEventType.CONTENT_BLOCK_STOP,
                index=event.index,
            )

        elif event_type == "message_delta":
            # Contains usage info and stop reason
            return StreamChunk(
                event_type=StreamEventType.MESSAGE_DELTA,
                usage={
                    "output_tokens": getattr(event.usage, "output_tokens", 0),
                },
            )

        return None
