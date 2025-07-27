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
from patterpunk.llm.types import ToolDefinition
from patterpunk.logger import logger


if anthropic:
    from anthropic import APIError


@dataclass
class ThinkingConfig:
    """Configuration for Claude reasoning mode."""
    type: Literal["enabled"] = "enabled"
    budget_tokens: int = 4000


class AnthropicRateLimitError(Exception):
    """Raised when all retry attempts for rate limit errors are exhausted"""

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
        """
        Convert OpenAI-style tool definitions to Anthropic format.

        OpenAI format: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        Anthropic format: {"name": "...", "description": "...", "input_schema": {...}}
        """
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

    def _parse_model_version(self) -> tuple[int, int]:
        """Parse major and minor version from model name. Returns (major, minor)."""
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
        """Check if the model supports reasoning mode (3.7+ or 4+)."""
        major, minor = self._parse_model_version()
        
        if major >= 4:
            return True
        elif major == 3 and minor >= 7:
            return True
            
        return False

    def _get_compatible_params(self, api_params: dict) -> dict:
        """Get parameters compatible with reasoning mode constraints."""
        if self.thinking:
            # Reasoning mode constraints:
            # - temperature must be exactly 1.0
            # - top_p and top_k must be removed
            compatible_params = api_params.copy()
            if "top_p" in compatible_params:
                del compatible_params["top_p"]
            if "top_k" in compatible_params:
                del compatible_params["top_k"]
            # Force temperature to 1.0 for reasoning mode
            compatible_params["temperature"] = 1.0
            return compatible_params
        return api_params

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
    ) -> Union[Message, "ToolCallMessage"]:
        system_prompt = "\n\n".join(
            [message.content for message in messages if message.role == ROLE_SYSTEM]
        )

        retry_count = 0
        wait_time = 60  # Initial wait time in seconds

        while True:
            try:
                # Prepare API parameters
                api_params = {
                    "model": self.model,
                    "system": system_prompt,
                    "messages": [
                        message.to_dict(prompt_for_structured_output=True)
                        for message in messages
                        if message.role in [ROLE_USER, ROLE_ASSISTANT]
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "timeout": self.timeout,
                }

                # Add thinking parameter for reasoning models
                if self.thinking and self._is_reasoning_model():
                    api_params["thinking"] = {
                        "type": self.thinking.type,
                        "budget_tokens": self.thinking.budget_tokens
                    }
                    # Apply reasoning mode parameter constraints
                    api_params = self._get_compatible_params(api_params)

                # Add tools if provided
                if tools:
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
                    for block in response.content:
                        if block.type == "tool_use":
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
