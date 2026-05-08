"""Unit tests for GoogleModel diagnostics (safety_settings, allow_empty_response,
finish_reason mapping, prompt_block_reason).

These tests do not hit the Vertex AI API. They use SimpleNamespace mocks for
response objects and exercise the synchronous helper functions directly.
"""

from copy import deepcopy
from types import SimpleNamespace
from typing import Optional

import pytest
from google.genai import types

from patterpunk.llm.finish_reason import FinishReason
from patterpunk.llm.models.google import (
    GoogleAPIError,
    GoogleModel,
    _build_all_safety_off,
    _extract_finish_reason_raw,
    _extract_prompt_block_reason,
    _normalize_finish_reason,
)

# ---- _build_all_safety_off ----


def test_build_all_safety_off_includes_known_text_categories():
    settings = _build_all_safety_off()
    categories = {s.category for s in settings}
    assert types.HarmCategory.HARM_CATEGORY_HARASSMENT in categories
    assert types.HarmCategory.HARM_CATEGORY_HATE_SPEECH in categories
    assert types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT in categories
    assert types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT in categories


def test_build_all_safety_off_excludes_unspecified():
    settings = _build_all_safety_off()
    categories = {s.category for s in settings}
    assert types.HarmCategory.HARM_CATEGORY_UNSPECIFIED not in categories


def test_build_all_safety_off_threshold_is_off():
    settings = _build_all_safety_off()
    assert all(s.threshold == types.HarmBlockThreshold.OFF for s in settings)


# ---- _normalize_finish_reason ----


def test_normalize_finish_reason_known_stop():
    assert _normalize_finish_reason("STOP") is FinishReason.STOP


def test_normalize_finish_reason_known_safety_variants():
    for raw in (
        "SAFETY",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
    ):
        assert _normalize_finish_reason(raw) is FinishReason.SAFETY


def test_normalize_finish_reason_unknown_falls_through():
    assert _normalize_finish_reason("RECITATION") is FinishReason.OTHER
    assert _normalize_finish_reason("MALFORMED_FUNCTION_CALL") is FinishReason.OTHER
    assert _normalize_finish_reason("FUTURE_NEW_VALUE") is FinishReason.OTHER


def test_normalize_finish_reason_none():
    assert _normalize_finish_reason(None) is None


# ---- _extract_finish_reason_raw ----


def _enum_like(name: str):
    """Mimic the SDK's enum-like object that has a .name attribute."""
    return SimpleNamespace(name=name)


def test_extract_finish_reason_from_response_with_candidates():
    response = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=_enum_like("STOP"))],
    )
    assert _extract_finish_reason_raw(response) == "STOP"


def test_extract_finish_reason_from_string_value():
    response = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason="STOP")],
    )
    assert _extract_finish_reason_raw(response) == "STOP"


def test_extract_finish_reason_returns_none_when_no_candidates():
    assert _extract_finish_reason_raw(SimpleNamespace(candidates=[])) is None
    assert _extract_finish_reason_raw(SimpleNamespace()) is None


def test_extract_finish_reason_returns_none_when_unset_on_candidate():
    response = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=None)],
    )
    assert _extract_finish_reason_raw(response) is None


# ---- _extract_prompt_block_reason ----


def test_extract_prompt_block_reason_from_feedback():
    response = SimpleNamespace(
        prompt_feedback=SimpleNamespace(block_reason=_enum_like("SAFETY")),
    )
    assert _extract_prompt_block_reason(response) == "SAFETY"


def test_extract_prompt_block_reason_returns_none_when_absent():
    assert _extract_prompt_block_reason(SimpleNamespace()) is None
    no_reason = SimpleNamespace(prompt_feedback=SimpleNamespace(block_reason=None))
    assert _extract_prompt_block_reason(no_reason) is None
    nested_none = SimpleNamespace(prompt_feedback=None)
    assert _extract_prompt_block_reason(nested_none) is None


# ---- safety_settings constructor + _build_generation_config ----


def _make_test_model(**kwargs) -> GoogleModel:
    """Construct a GoogleModel with a fake client to avoid auth in unit tests."""
    fake_client = SimpleNamespace()
    return GoogleModel(model="gemini-2.5-pro", client=fake_client, **kwargs)


def test_safety_settings_default_omits_field():
    model = _make_test_model()
    config = model._build_generation_config(
        tools=None,
        structured_output=None,
        output_types=None,
        system_instruction=None,
    )
    # Pydantic models initialize unset Optional fields to None.
    assert getattr(config, "safety_settings", None) is None


def test_safety_settings_passed_through():
    settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.OFF,
        ),
    ]
    model = _make_test_model(safety_settings=settings)
    config = model._build_generation_config(
        tools=None,
        structured_output=None,
        output_types=None,
        system_instruction=None,
    )
    assert config.safety_settings == settings


def test_disable_safety_filters_applies_all_off_when_no_explicit_settings():
    model = _make_test_model()
    config = model._build_generation_config(
        tools=None,
        structured_output=None,
        output_types=None,
        system_instruction=None,
        disable_safety_filters=True,
    )
    assert config.safety_settings is not None
    assert all(
        s.threshold == types.HarmBlockThreshold.OFF for s in config.safety_settings
    )


def test_explicit_safety_settings_override_disable_safety_filters():
    """Model-level safety_settings is more specific than chat-level disable flag."""
    explicit = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ]
    model = _make_test_model(safety_settings=explicit)
    config = model._build_generation_config(
        tools=None,
        structured_output=None,
        output_types=None,
        system_instruction=None,
        disable_safety_filters=True,
    )
    assert config.safety_settings == explicit


def test_safety_settings_survives_deepcopy():
    settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.OFF,
        ),
    ]
    model = _make_test_model(safety_settings=settings, allow_empty_response=True)
    cloned = deepcopy(model)
    assert cloned.safety_settings == settings
    assert cloned.allow_empty_response is True


# ---- allow_empty_response + _process_generation_response ----


def _empty_response(
    finish_reason_name: Optional[str] = None, prompt_block_name: Optional[str] = None
):
    """Construct a response shape that triggers the empty-response path."""
    return SimpleNamespace(
        candidates=[],
        prompt_feedback=(
            SimpleNamespace(block_reason=_enum_like(prompt_block_name))
            if prompt_block_name is not None
            else None
        ),
    )


def test_empty_response_raises_by_default():
    model = _make_test_model()
    with pytest.raises(GoogleAPIError):
        model._process_generation_response(_empty_response(), structured_output=None)


def test_empty_response_returns_empty_message_when_allowed():
    model = _make_test_model(allow_empty_response=True)
    result = model._process_generation_response(
        _empty_response(), structured_output=None
    )
    assert result.content == ""
    assert result.finish_reason is None
    assert result._provider.raw_finish_reason is None
    assert result._provider.prompt_block_reason is None


def test_empty_response_preserves_diagnostics_when_allowed():
    """Even on empty content, finish_reason and prompt_block_reason carry through."""
    model = _make_test_model(allow_empty_response=True)
    response = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=_enum_like("SAFETY"))],
        prompt_feedback=SimpleNamespace(block_reason=_enum_like("OTHER")),
    )
    result = model._process_generation_response(response, structured_output=None)
    assert result.content == ""
    assert result.finish_reason is FinishReason.SAFETY
    assert result._provider.raw_finish_reason == "SAFETY"
    assert result._provider.prompt_block_reason == "OTHER"


def test_normal_response_carries_diagnostics():
    """A successful response also populates finish_reason + raw_finish_reason."""
    model = _make_test_model()
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason=_enum_like("STOP"),
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text="hello world",
                            function_call=None,
                            inline_data=None,
                            thought=False,
                        )
                    ]
                ),
            )
        ],
        prompt_feedback=None,
        usage_metadata=None,
    )
    result = model._process_generation_response(response, structured_output=None)
    assert result.content == "hello world"
    assert result.finish_reason is FinishReason.STOP
    assert result._provider.raw_finish_reason == "STOP"
    assert result._provider.prompt_block_reason is None
