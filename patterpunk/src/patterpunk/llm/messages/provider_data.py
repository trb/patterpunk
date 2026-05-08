"""Provider-specific message metadata escape hatch.

Holds fields that don't map cleanly to the unified ``AssistantMessage``
interface. Access fields via attribute syntax; missing fields return ``None``.

Per-provider populated fields:
    Google:    raw_finish_reason, prompt_block_reason
    Anthropic: raw_finish_reason
    OpenAI:    raw_finish_reason
    Bedrock:   raw_finish_reason
    Ollama:    raw_finish_reason
"""

from typing import Optional


class ProviderData:
    def __init__(self, **fields):
        self._fields = dict(fields)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._fields.get(name)

    def __repr__(self) -> str:
        return f"ProviderData({self._fields!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProviderData):
            return NotImplemented
        return self._fields == other._fields

    def __bool__(self) -> bool:
        return bool(self._fields)

    def to_dict(self) -> dict:
        return dict(self._fields)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ProviderData":
        return cls(**(data or {}))
