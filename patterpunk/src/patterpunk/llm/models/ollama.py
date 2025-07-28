import json
import random
from abc import ABC
from typing import List, Optional, Callable, Union

from patterpunk.config import ollama
from patterpunk.lib.structured_output import get_model_schema, has_model_schema
from patterpunk.llm.messages import Message, AssistantMessage, ToolCallMessage
from patterpunk.llm.models.base import Model
from patterpunk.llm.types import ToolDefinition, CacheChunk


class OllamaModel(Model, ABC):
    # @todo: Add thinking mode support when Ollama supports reasoning models
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

    def _convert_tools_to_ollama_format(self, tools: ToolDefinition) -> List[dict]:
        """Convert Patterpunk standard tools to Ollama format"""
        ollama_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                # Ollama uses the same format as OpenAI, so minimal conversion needed
                ollama_tools.append(tool)
        
        return ollama_tools

    def _convert_messages_for_ollama(self, messages: List[Message]) -> List[dict]:
        """Convert patterpunk messages to Ollama format, ignoring cache settings."""
        ollama_messages = []
        
        for message in messages:
            # Always convert to string, ignoring cache settings
            content = message.get_content_as_string()
            
            ollama_messages.append({
                "role": message.role,
                "content": content
            })
        
        return ollama_messages

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
    ) -> Union[Message, "ToolCallMessage"]:
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

        chat_params = {
            "model": self.model,
            "messages": self._convert_messages_for_ollama(messages),
            "stream": False,
            "format": (
                get_model_schema(structured_output)
                if has_model_schema(structured_output)
                else None
            ),
            "options": options,
        }

        # Add tools if provided
        if tools:
            ollama_tools = self._convert_tools_to_ollama_format(tools)
            if ollama_tools:
                chat_params["tools"] = ollama_tools

        response = ollama.chat(**chat_params)

        # Check for tool calls first
        if response.get("message", {}).get("tool_calls"):
            tool_calls = []
            for tool_call in response["message"]["tool_calls"]:
                # Ensure tool call has an ID, generate one if missing
                call_id = tool_call.get("id")
                if not call_id:
                    call_id = f"call_{tool_call['function']['name']}_{random.randint(1000, 9999)}"
                
                formatted_tool_call = {
                    "id": call_id,
                    "type": tool_call.get("type", "function"),
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"]
                    }
                }
                tool_calls.append(formatted_tool_call)
            
            if tool_calls:
                return ToolCallMessage(tool_calls)

        # If no tool calls, return regular assistant message
        return AssistantMessage(
            response["message"]["content"], structured_output=structured_output
        )

    @staticmethod
    def get_name():
        return "Ollama"

    @staticmethod
    def get_available_models() -> List[str]:
        return [model["model"] for model in ollama.list()["models"]]
