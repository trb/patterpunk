"""Per-provider finish_reason normalization tests.

Each provider has its own translation table. These tests ensure each table
maps known values correctly, falls through to OTHER for unknown values, and
that ProviderData.raw_finish_reason carries the provider-native string.

All tests use SimpleNamespace mocks — no API calls.
"""

from types import SimpleNamespace

from patterpunk.llm.finish_reason import FinishReason
from patterpunk.llm.models.anthropic import (
    _build_diagnostics_kwargs as _anthropic_build_diagnostics,
    _normalize_finish_reason as _anthropic_normalize,
)
from patterpunk.llm.models.bedrock import (
    _build_diagnostics_kwargs as _bedrock_build_diagnostics,
    _normalize_finish_reason as _bedrock_normalize,
)
from patterpunk.llm.models.ollama import (
    _build_diagnostics_kwargs as _ollama_build_diagnostics,
    _normalize_finish_reason as _ollama_normalize,
)
from patterpunk.llm.models.openai import (
    _extract_raw_finish_reason as _openai_extract_raw,
    _normalize_finish_reason as _openai_normalize,
)

# ============================================================================
# Anthropic
# ============================================================================


def test_anthropic_normalize_known_values():
    assert _anthropic_normalize("end_turn") is FinishReason.STOP
    assert _anthropic_normalize("stop_sequence") is FinishReason.STOP
    assert _anthropic_normalize("max_tokens") is FinishReason.MAX_TOKENS
    assert _anthropic_normalize("tool_use") is FinishReason.TOOL_USE
    assert _anthropic_normalize("refusal") is FinishReason.SAFETY


def test_anthropic_normalize_unknown_falls_through_to_other():
    assert _anthropic_normalize("model_context_window_exceeded") is FinishReason.OTHER
    assert _anthropic_normalize("future_value") is FinishReason.OTHER


def test_anthropic_normalize_none():
    assert _anthropic_normalize(None) is None


def test_anthropic_diagnostics_kwargs():
    response = SimpleNamespace(stop_reason="end_turn")
    kwargs = _anthropic_build_diagnostics(response)
    assert kwargs["finish_reason"] is FinishReason.STOP
    assert kwargs["provider_data"].raw_finish_reason == "end_turn"


def test_anthropic_diagnostics_kwargs_handles_missing_stop_reason():
    kwargs = _anthropic_build_diagnostics(SimpleNamespace())
    assert kwargs["finish_reason"] is None
    assert kwargs["provider_data"].raw_finish_reason is None


# ============================================================================
# OpenAI
# ============================================================================


def test_openai_normalize_completed():
    assert _openai_normalize("completed") is FinishReason.STOP


def test_openai_normalize_max_output_tokens_and_content_filter():
    assert _openai_normalize("max_output_tokens") is FinishReason.MAX_TOKENS
    assert _openai_normalize("content_filter") is FinishReason.SAFETY


def test_openai_normalize_failed_falls_to_other():
    assert _openai_normalize("failed") is FinishReason.OTHER
    assert _openai_normalize("cancelled") is FinishReason.OTHER


def test_openai_extract_uses_incomplete_reason_when_status_is_incomplete():
    response = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="content_filter"),
    )
    assert _openai_extract_raw(response) == "content_filter"


def test_openai_extract_uses_status_when_completed():
    response = SimpleNamespace(status="completed", incomplete_details=None)
    assert _openai_extract_raw(response) == "completed"


def test_openai_extract_handles_missing_status():
    assert _openai_extract_raw(SimpleNamespace()) is None


# ============================================================================
# Bedrock
# ============================================================================


def test_bedrock_normalize_known_values():
    assert _bedrock_normalize("end_turn") is FinishReason.STOP
    assert _bedrock_normalize("stop_sequence") is FinishReason.STOP
    assert _bedrock_normalize("max_tokens") is FinishReason.MAX_TOKENS
    assert _bedrock_normalize("tool_use") is FinishReason.TOOL_USE
    assert _bedrock_normalize("guardrail_intervened") is FinishReason.SAFETY
    assert _bedrock_normalize("content_filtered") is FinishReason.SAFETY


def test_bedrock_normalize_unknown_falls_through():
    assert _bedrock_normalize("malformed_model_output") is FinishReason.OTHER
    assert _bedrock_normalize("malformed_tool_use") is FinishReason.OTHER
    assert _bedrock_normalize("model_context_window_exceeded") is FinishReason.OTHER


def test_bedrock_diagnostics_kwargs_carries_raw():
    kwargs = _bedrock_build_diagnostics("guardrail_intervened")
    assert kwargs["finish_reason"] is FinishReason.SAFETY
    assert kwargs["provider_data"].raw_finish_reason == "guardrail_intervened"


# ============================================================================
# Ollama
# ============================================================================


def test_ollama_normalize_stop_and_length():
    assert _ollama_normalize("stop") is FinishReason.STOP
    assert _ollama_normalize("length") is FinishReason.MAX_TOKENS


def test_ollama_normalize_load_unload_falls_through():
    assert _ollama_normalize("load") is FinishReason.OTHER
    assert _ollama_normalize("unload") is FinishReason.OTHER


def test_ollama_diagnostics_kwargs_from_response_dict():
    kwargs = _ollama_build_diagnostics({"done_reason": "stop"})
    assert kwargs["finish_reason"] is FinishReason.STOP
    assert kwargs["provider_data"].raw_finish_reason == "stop"


def test_ollama_diagnostics_kwargs_handles_missing_done_reason():
    kwargs = _ollama_build_diagnostics({})
    assert kwargs["finish_reason"] is None
    assert kwargs["provider_data"].raw_finish_reason is None
