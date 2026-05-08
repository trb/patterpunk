"""Cross-cutting tests for FinishReason, ProviderData, and AssistantMessage diagnostics.

These tests exercise pure-Python data types and don't require provider APIs.
"""

from types import SimpleNamespace
from typing import List, Optional, Set, Union

import pytest

from patterpunk.llm.chat.core import Chat
from patterpunk.llm.finish_reason import FinishReason
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.base import Message
from patterpunk.llm.messages.provider_data import ProviderData
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.base import Model
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.types import ToolDefinition

# ---- FinishReason enum ----


def test_finish_reason_str_equality():
    assert FinishReason.STOP == "stop"
    assert FinishReason.MAX_TOKENS == "max_tokens"
    assert FinishReason.TOOL_USE == "tool_use"
    assert FinishReason.SAFETY == "safety"
    assert FinishReason.OTHER == "other"


def test_finish_reason_reverse_lookup():
    assert FinishReason("stop") is FinishReason.STOP
    assert FinishReason("safety") is FinishReason.SAFETY


def test_finish_reason_unknown_raises():
    with pytest.raises(ValueError):
        FinishReason("garbage_value")


# ---- ProviderData ----


def test_provider_data_default_returns_none_for_missing():
    data = ProviderData()
    assert data.raw_finish_reason is None
    assert data.prompt_block_reason is None
    assert data.anything_at_all is None


def test_provider_data_returns_set_fields():
    data = ProviderData(raw_finish_reason="STOP", prompt_block_reason=None)
    assert data.raw_finish_reason == "STOP"
    assert data.prompt_block_reason is None


def test_provider_data_round_trip():
    original = ProviderData(raw_finish_reason="end_turn", prompt_block_reason="SAFETY")
    restored = ProviderData.from_dict(original.to_dict())
    assert restored == original
    assert restored.raw_finish_reason == "end_turn"
    assert restored.prompt_block_reason == "SAFETY"


def test_provider_data_truthiness():
    assert not ProviderData()
    assert ProviderData(raw_finish_reason="end_turn")


def test_provider_data_from_dict_handles_none():
    data = ProviderData.from_dict(None)
    assert data == ProviderData()
    assert data.raw_finish_reason is None


# ---- AssistantMessage diagnostics ----


def test_assistant_message_default_no_diagnostics():
    msg = AssistantMessage("hi")
    assert msg.finish_reason is None
    assert isinstance(msg._provider, ProviderData)
    assert msg._provider.raw_finish_reason is None


def test_assistant_message_with_finish_reason():
    msg = AssistantMessage("hi", finish_reason=FinishReason.STOP)
    assert msg.finish_reason == FinishReason.STOP
    assert msg.finish_reason == "stop"  # str-Enum equality


def test_assistant_message_serialize_omits_diagnostics_when_unset():
    msg = AssistantMessage("hi")
    serialized = msg.serialize()
    assert "finish_reason" not in serialized
    assert "provider_data" not in serialized


def test_assistant_message_serialize_includes_finish_reason_when_set():
    msg = AssistantMessage("hi", finish_reason=FinishReason.MAX_TOKENS)
    serialized = msg.serialize()
    assert serialized["finish_reason"] == "max_tokens"


def test_assistant_message_serialize_includes_provider_data_when_populated():
    msg = AssistantMessage(
        "hi",
        provider_data=ProviderData(raw_finish_reason="end_turn"),
    )
    serialized = msg.serialize()
    assert serialized["provider_data"] == {"raw_finish_reason": "end_turn"}


def test_assistant_message_round_trip_preserves_diagnostics():
    msg = AssistantMessage(
        "hi",
        finish_reason=FinishReason.SAFETY,
        provider_data=ProviderData(
            raw_finish_reason="content_filter",
            prompt_block_reason=None,
        ),
    )
    restored = AssistantMessage.deserialize(msg.serialize())
    assert restored.finish_reason == FinishReason.SAFETY
    assert restored._provider.raw_finish_reason == "content_filter"


def test_assistant_message_round_trip_legacy_payload_without_diagnostics():
    """A serialized message from before this feature must still deserialize cleanly."""
    legacy_payload = {
        "type": "assistant",
        "id": "msg_legacy",
        "content": "hi",
    }
    restored = AssistantMessage.deserialize(legacy_payload)
    assert restored.content == "hi"
    assert restored.finish_reason is None
    assert restored._provider.raw_finish_reason is None


def test_assistant_message_deserialize_corrupt_finish_reason_raises():
    """An unknown finish_reason value should fail loudly at the boundary."""
    payload = {
        "type": "assistant",
        "id": "msg_corrupt",
        "content": "hi",
        "finish_reason": "garbage_value",
    }
    with pytest.raises(ValueError):
        AssistantMessage.deserialize(payload)


# ---- Chat plumbing ----


class _RecordingModel(Model):
    """Test fake that records the kwargs Chat passes when calling generate_assistant_message.

    Overrides ``__deepcopy__`` to return self so the test holds a reference to
    the same instance that Chat uses after copy-on-write chaining.
    """

    def __init__(self):
        self.last_kwargs: Optional[dict] = None

    def __deepcopy__(self, memo):
        return self

    def generate_assistant_message(
        self,
        messages: List[Message],
        tools: Optional[ToolDefinition] = None,
        structured_output: Optional[object] = None,
        output_types: Optional[Union[List[OutputType], Set[OutputType]]] = None,
        disable_safety_filters: bool = False,
    ):
        self.last_kwargs = {
            "tools": tools,
            "structured_output": structured_output,
            "output_types": output_types,
            "disable_safety_filters": disable_safety_filters,
        }
        return AssistantMessage("test-response")

    @staticmethod
    def get_name() -> str:
        return "_RecordingModel"

    @staticmethod
    def get_available_models() -> List[str]:
        return []

    def count_tokens(self, content) -> int:
        return 0


def test_chat_default_disable_safety_filters_is_false():
    fake_model = _RecordingModel()
    chat = Chat(model=fake_model)
    chat.add_message(UserMessage("hello")).complete()
    assert fake_model.last_kwargs["disable_safety_filters"] is False


def test_chat_passes_disable_safety_filters_to_model():
    fake_model = _RecordingModel()
    chat = Chat(model=fake_model, disable_safety_filters=True)
    chat.add_message(UserMessage("hello")).complete()
    assert fake_model.last_kwargs["disable_safety_filters"] is True


def test_chat_disable_safety_filters_propagates_through_chain():
    """Chained methods (add_message, etc.) preserve disable_safety_filters."""
    fake_model = _RecordingModel()
    chat = (
        Chat(model=fake_model, disable_safety_filters=True)
        .add_message(UserMessage("hello"))
        .add_message(UserMessage("again"))
    )
    chat.complete()
    assert fake_model.last_kwargs["disable_safety_filters"] is True
