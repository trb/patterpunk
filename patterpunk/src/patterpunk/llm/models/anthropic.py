import json
import time
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Callable, get_args, Union, Literal

from patterpunk.config import (
    anthropic,
    ANTHROPIC_DEFAULT_TEMPERATURE,
    ANTHROPIC_DEFAULT_TOP_P,
    ANTHROPIC_DEFAULT_TOP_K,
    ANTHROPIC_DEFAULT_MAX_TOKENS,
    ANTHROPIC_DEFAULT_TIMEOUT,
    MAX_RETRIES,
)
from patterpunk.llm.messages import (
    Message,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    AssistantMessage,
    ToolCallMessage,
)
from patterpunk.llm.models.base import Model
from patterpunk.llm.thinking import ThinkingConfig as UnifiedThinkingConfig
from patterpunk.llm.types import ToolDefinition, CacheChunk
from patterpunk.llm.multimodal import MultimodalChunk
from patterpunk.llm.text import TextChunk
from patterpunk.llm.messages import get_multimodal_chunks, has_multimodal_content
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
                effort_to_tokens = {
                    "low": 2000,
                    "medium": 8000, 
                    "high": 24000
                }
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
            raise ValueError("structured_output must be a Pydantic model with schema support")
        
        schema = get_model_schema(structured_output)
        
        return {
            "name": "provide_structured_response",
            "description": "Provide the response in the exact structured format specified by the schema.",
            "input_schema": schema
        }

    def _format_reasoning_to_structured_output(self, reasoning_content: str, structured_output: object, original_messages: List[Message]) -> AssistantMessage:
        """
        Reasoning models can't use tool_choice constraints, so we use a two-model approach:
        first the reasoning model generates analysis, then Haiku formats it to structured JSON.
        We use Haiku because it's fast, cheap, and reliable for formatting tasks.
        """
        logger.info("[ANTHROPIC] Formatting reasoning output to structured JSON using Claude 3.5 Haiku")
        
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
            "tool_choice": {
                "type": "tool", 
                "name": "provide_structured_response"
            }
        }
        
        try:
            formatting_response = anthropic.messages.create(**haiku_params)
            
            for block in formatting_response.content:
                if block.type == "tool_use" and block.name == "provide_structured_response":
                    if hasattr(block, "input") and block.input:
                        try:
                            parsed_output = structured_output.model_validate(block.input)
                            structured_response_content = json.dumps(block.input, indent=2)
                            
                            logger.info("[ANTHROPIC] Successfully formatted reasoning output to structured JSON")
                            return AssistantMessage(
                                structured_response_content,
                                structured_output=structured_output,
                                parsed_output=parsed_output
                            )
                        except Exception as e:
                            logger.error(f"[ANTHROPIC] Failed to parse structured output from Haiku formatting: {e}")
                            
            logger.warning("[ANTHROPIC] Haiku formatting failed, falling back to reasoning content")
            return AssistantMessage(reasoning_content, structured_output=structured_output)
            
        except Exception as e:
            logger.error(f"[ANTHROPIC] Error in two-model structured output approach: {e}")
            return AssistantMessage(reasoning_content, structured_output=structured_output)

    def _parse_model_version(self) -> tuple[int, int]:
        import re
        
        # Claude 3.x format: claude-3-7-sonnet-20250219
        claude3_match = re.search(r'claude-3-(\d+)', self.model)
        if claude3_match:
            return (3, int(claude3_match.group(1)))
            
        # Claude 4+ format: claude-opus-4-20250514, claude-sonnet-4-5-20250614
        # Need to distinguish between minor version and date
        claude4plus_match = re.search(r'claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d+))?-(\d{8})', self.model)
        if claude4plus_match:
            major = int(claude4plus_match.group(1))
            # Group 2 is minor version, Group 3 is date (8 digits)
            minor_str = claude4plus_match.group(2)
            minor = int(minor_str) if minor_str else 0
            return (major, minor)
            
        return (0, 0)  # Unknown/unsupported format

    def _is_reasoning_model(self) -> bool:
        major, minor = self._parse_model_version()
        
        if major >= 4:
            return True
        elif major == 3 and minor >= 7:
            return True
            
        return False

    def _get_compatible_params(self, api_params: dict) -> dict:
        if self.thinking:
            compatible_params = api_params.copy()
            if "top_p" in compatible_params:
                del compatible_params["top_p"]
            if "top_k" in compatible_params:
                del compatible_params["top_k"]
            compatible_params["temperature"] = 1.0
            return compatible_params
        return api_params

    def _convert_content_to_anthropic_format(self, content) -> List[dict]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        
        anthropic_content = []
        session = None
        
        for chunk in content:
            if isinstance(chunk, TextChunk):
                anthropic_content.append({
                    "type": "text",
                    "text": chunk.content
                })
                
            elif isinstance(chunk, CacheChunk):
                content_block = {
                    "type": "text",
                    "text": chunk.content
                }
                
                if chunk.cacheable:
                    cache_control = {"type": "ephemeral"}
                    if chunk.ttl:
                        cache_control["ttl"] = int(chunk.ttl.total_seconds())
                    content_block["cache_control"] = cache_control
                
                anthropic_content.append(content_block)
                
            elif isinstance(chunk, MultimodalChunk):
                if hasattr(chunk, 'file_id'):
                    content_block = {
                        "type": "document",
                        "source": {
                            "type": "file",
                            "file_id": chunk.file_id
                        }
                    }
                    anthropic_content.append(content_block)
                    continue
                
                if chunk.source_type == "url":
                    if session is None:
                        try:
                            import requests
                            session = requests.Session()
                        except ImportError:
                            raise ImportError("requests library required for URL support with Anthropic")
                    
                    chunk = chunk.download(session)
                
                media_type = chunk.media_type or "application/octet-stream"
                
                if media_type.startswith("image/"):
                    content_block = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": chunk.to_base64()
                        }
                    }
                    anthropic_content.append(content_block)
                elif media_type == "application/pdf":
                    content_block = {
                        "type": "document", 
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": chunk.to_base64()
                        }
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
                delete=False
            ) as tmp_file:
                tmp_file.write(chunk.to_bytes())
                tmp_file_path = tmp_file.name
            
            try:
                with open(tmp_file_path, "rb") as f:
                    file_response = client.files.create(
                        file=f,
                        purpose="vision"
                    )
                
                return file_response.id
            finally:
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"Failed to upload file to Anthropic: {e}")
            raise

    def _convert_messages_for_anthropic(self, messages: List[Message]) -> List[dict]:
        anthropic_messages = []
        
        for message in messages:
            if isinstance(message.content, list):
                content = self._convert_content_to_anthropic_format(message.content)
            else:
                content = [{"type": "text", "text": message.content}]
            
            anthropic_messages.append({
                "role": message.role,
                "content": content
            })
        
        return anthropic_messages

    def _convert_system_messages_for_anthropic(self, messages: List[Message]) -> List[dict]:
        system_content = []
        
        for message in messages:
            if message.role == ROLE_SYSTEM:
                if isinstance(message.content, list):
                    system_content.extend(self._convert_content_to_anthropic_format(message.content))
                else:
                    system_content.append({"type": "text", "text": message.content})
        
        return system_content

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
    ) -> Union[Message, "ToolCallMessage"]:
        system_content = self._convert_system_messages_for_anthropic(messages)
        system_prompt = None
        if system_content:
            if len(system_content) == 1 and system_content[0].get("type") == "text":
                if "cache_control" not in system_content[0]:
                    system_prompt = system_content[0]["text"]
                else:
                    # Has cache controls, use content array format
                    system_prompt = system_content
            else:
                # Multiple blocks or complex content, use content array format
                system_prompt = system_content

        retry_count = 0
        wait_time = 60  # Initial wait time in seconds

        while True:
            try:
                api_params = {
                    "model": self.model,
                    "messages": self._convert_messages_for_anthropic([
                        message for message in messages
                        if message.role in [ROLE_USER, ROLE_ASSISTANT]
                    ]),
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "timeout": self.timeout,
                }

                if system_prompt is not None:
                    api_params["system"] = system_prompt

                if self.thinking and self._is_reasoning_model():
                    api_params["thinking"] = {
                        "type": self.thinking.type,
                        "budget_tokens": self.thinking.budget_tokens
                    }
                    api_params = self._get_compatible_params(api_params)

                if structured_output and has_model_schema(structured_output) and self.thinking and self._is_reasoning_model():
                    # Try reasoning model with auto tool choice first (compatible with thinking)
                    logger.info(f"[ANTHROPIC] Attempting reasoning model with auto tool choice for structured output")
                    
                    structured_output_tool = self._create_structured_output_tool(structured_output)
                    
                    anthropic_tools = []
                    if tools:
                        anthropic_tools = self._convert_tools_to_anthropic_format(tools)
                    
                    # Add the structured output tool
                    anthropic_tools.append(structured_output_tool)
                    api_params["tools"] = anthropic_tools
                    
                    # Use auto tool choice (compatible with reasoning mode)
                    api_params["tool_choice"] = {"type": "auto"}
                    
                    # Make the reasoning call with auto tool choice
                    try:
                        reasoning_response = anthropic.messages.create(**api_params)
                        
                        # Check if the model used the structured output tool
                        for block in reasoning_response.content:
                            if block.type == "tool_use" and block.name == "provide_structured_response":
                                if hasattr(block, "input") and block.input:
                                    try:
                                        # Parse the structured output
                                        parsed_output = structured_output.model_validate(block.input)
                                        # Create content from the structured data
                                        structured_response_content = json.dumps(block.input, indent=2)
                                        
                                        logger.info("[ANTHROPIC] Successfully got structured output from reasoning model with auto tool choice")
                                        return AssistantMessage(
                                            structured_response_content,
                                            structured_output=structured_output,
                                            parsed_output=parsed_output
                                        )
                                    except Exception as e:
                                        logger.warning(f"[ANTHROPIC] Failed to parse structured output from reasoning model: {e}")
                                        break
                        
                        # If we get here, the reasoning model didn't use the structured output tool
                        # Fall back to two-model approach
                        logger.info("[ANTHROPIC] Reasoning model didn't use structured output tool, falling back to two-model approach")
                        
                        # Extract the reasoning content (handle both text and tool calls)
                        reasoning_parts = []
                        for block in reasoning_response.content:
                            if block.type == "text":
                                reasoning_parts.append(block.text)
                            elif block.type == "tool_use" and block.name != "provide_structured_response":
                                # Include other tool calls in the reasoning context
                                reasoning_parts.append(f"Tool call: {block.name} with arguments: {json.dumps(block.input)}")
                        
                        reasoning_content = "\n".join(reasoning_parts)
                        
                        # Second call: use Claude 3.5 Haiku to format the reasoning into structured output
                        return self._format_reasoning_to_structured_output(
                            reasoning_content, structured_output, messages
                        )
                        
                    except Exception as e:
                        logger.warning(f"[ANTHROPIC] Error with reasoning model auto tool choice: {e}, falling back to two-model approach")
                        
                        # Fall back to two-model approach without structured output tool
                        if tools:
                            anthropic_tools = self._convert_tools_to_anthropic_format(tools)
                            api_params["tools"] = anthropic_tools
                            api_params["tool_choice"] = {"type": "auto"}
                        
                        # Make the reasoning call
                        reasoning_response = anthropic.messages.create(**api_params)
                        
                        # Extract the reasoning content
                        reasoning_parts = []
                        for block in reasoning_response.content:
                            if block.type == "text":
                                reasoning_parts.append(block.text)
                            elif block.type == "tool_use":
                                reasoning_parts.append(f"Tool call: {block.name} with arguments: {json.dumps(block.input)}")
                        
                        reasoning_content = "\n".join(reasoning_parts)
                        
                        # Second call: use Claude 3.5 Haiku to format the reasoning into structured output
                        return self._format_reasoning_to_structured_output(
                            reasoning_content, structured_output, messages
                        )
                    
                elif structured_output and has_model_schema(structured_output):
                    # Regular structured output approach for non-reasoning models
                    structured_output_tool = self._create_structured_output_tool(structured_output)
                    
                    anthropic_tools = []
                    if tools:
                        anthropic_tools = self._convert_tools_to_anthropic_format(tools)
                    
                    # Add the structured output tool
                    anthropic_tools.append(structured_output_tool)
                    api_params["tools"] = anthropic_tools
                    
                    # Force the model to use the structured output tool
                    api_params["tool_choice"] = {
                        "type": "tool",
                        "name": "provide_structured_response"
                    }
                elif tools:
                    # Regular tool handling when no structured output
                    anthropic_tools = self._convert_tools_to_anthropic_format(tools)
                    if anthropic_tools:
                        api_params["tools"] = anthropic_tools

                response = anthropic.messages.create(**api_params)

                if response.stop_reason in ["end_turn", "stop_sequence", "max_tokens", "refusal"]:
                    if response.stop_reason == "max_tokens":
                        logger.warning(
                            "Anthropic response was cut off as the model hit MAX_TOKENS"
                        )
                    elif response.stop_reason == "refusal":
                        logger.warning(
                            "Anthropic model refused to generate content for safety reasons"
                        )
                    content = "\n".join(
                        [
                            block.text
                            for block in response.content
                            if block.type == "text"
                        ]
                    )
                    return AssistantMessage(
                        content, structured_output=structured_output
                    )
                elif response.stop_reason == "tool_use":
                    # Handle tool use response
                    tool_calls = []
                    structured_response_content = None
                    
                    for block in response.content:
                        if block.type == "tool_use":
                            # Check if this is our structured output tool
                            if block.name == "provide_structured_response" and structured_output:
                                # Extract the structured output directly from the tool call
                                if hasattr(block, "input") and block.input:
                                    try:
                                        # Parse the structured output
                                        parsed_output = structured_output.model_validate(block.input)
                                        # Create content from the structured data
                                        structured_response_content = json.dumps(block.input, indent=2)
                                        
                                        return AssistantMessage(
                                            structured_response_content,
                                            structured_output=structured_output,
                                            parsed_output=parsed_output
                                        )
                                    except Exception as e:
                                        logger.warning(f"Failed to parse structured output from tool call: {e}")
                                        # Fall back to regular tool call handling
                            
                            # Regular tool call handling
                            # Convert Anthropic tool use format to patterpunk standard format
                            # Anthropic returns input as a dict, we need to convert to JSON string
                            arguments = "{}"
                            if hasattr(block, "input") and block.input:
                                try:
                                    arguments = json.dumps(block.input)
                                except (TypeError, ValueError):
                                    arguments = str(block.input)

                            tool_call = {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": arguments,
                                },
                            }
                            tool_calls.append(tool_call)

                    if tool_calls:
                        return ToolCallMessage(tool_calls)
                    else:
                        raise AnthropicAPIError(
                            "Tool use stop reason but no tool use blocks found in response"
                        )
                else:
                    raise AnthropicAPIError(
                        f"Unknown stop reason: {response.stop_reason}"
                    )

            except APIError as e:
                if (
                    getattr(e, "status_code", None) == 429
                    or "rate_limit_error" in str(e).lower()
                ):
                    if retry_count >= MAX_RETRIES:
                        raise AnthropicRateLimitError(
                            f"Rate limit exceeded after {retry_count} retries"
                        ) from e

                    logger.warning(
                        f"Rate limit hit, attempt {retry_count + 1}/{MAX_RETRIES}. "
                        f"Waiting {wait_time} seconds before retry."
                    )

                    time.sleep(wait_time)
                    retry_count += 1
                    wait_time = int(wait_time * 1.5)  # Increase wait time by 50%
                    continue

                raise  # Re-raise any other API errors
        raise AnthropicAPIError(
            f"Unexpected outcome - out of retries, but neither error raised or message returned"
        )

    @staticmethod
    def get_available_models() -> List[str]:
        return [model.id for model in anthropic.models.list()]

    @staticmethod
    def get_name():
        return "Anthropic"
