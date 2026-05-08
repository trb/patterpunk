"""Normalized cross-provider finish-reason vocabulary.

Each provider model maps its native stop-reason enum to one of these values so
consumers can branch on a unified vocabulary without knowing which provider
produced the message.

For provider-native values (e.g. Google's ``RECITATION`` or Anthropic's
``stop_sequence``), see ``AssistantMessage._provider.raw_finish_reason``.

Note: ``TOOL_USE`` only appears on an ``AssistantMessage`` from Anthropic,
OpenAI, or Bedrock when the model emits both content and a tool call. Patterpunk
routes pure tool calls to ``ToolCallMessage`` (a separate type), and Google /
Ollama do not signal tool calls via the finish-reason field.
"""

from enum import StrEnum


class FinishReason(StrEnum):
    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    SAFETY = "safety"
    OTHER = "other"
