import json
import time
from abc import ABC
from typing import List, Optional, Callable, get_args, Union

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
from patterpunk.llm.types import ToolDefinition
from patterpunk.logger import logger


if anthropic:
    from anthropic import APIError


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
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.timeout = timeout

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
                    "input_schema": func["parameters"]
                }
                anthropic_tools.append(anthropic_tool)
        return anthropic_tools

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
                
                # Add tools if provided
                if tools:
                    anthropic_tools = self._convert_tools_to_anthropic_format(tools)
                    if anthropic_tools:
                        api_params["tools"] = anthropic_tools

                response = anthropic.messages.create(**api_params)

                if response.stop_reason in ["end_turn", "stop_sequence", "max_tokens"]:
                    if response.stop_reason == "max_tokens":
                        logger.warning(
                            "Anthropic response was cut off as the model hit MAX_TOKENS"
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
                            if hasattr(block, 'input') and block.input:
                                try:
                                    arguments = json.dumps(block.input)
                                except (TypeError, ValueError):
                                    arguments = str(block.input)
                            
                            tool_call = {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": arguments
                                }
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
