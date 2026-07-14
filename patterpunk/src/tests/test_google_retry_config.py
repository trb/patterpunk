"""Unit tests for GoogleModel retry behavior, both with a RetryConfig (the
schedule-driven path) and without one (the legacy env-driven 429-only loop).

These tests do not hit the Vertex AI API. The GoogleModel is constructed with
a Mock client (constructor injection skips auth) and errors are raised as real
genai APIError instances so the except clauses classify them exactly like
production traffic.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from google.genai import errors as genai_errors

from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.google import (
    GoogleAPIError,
    GoogleModel,
    GoogleRateLimitError,
)
from patterpunk.llm.retry_config import RetryConfig


def _api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(
        code, {"error": {"message": f"simulated error {code}"}}
    )


def _good_response(text: str = "hello world") -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason=SimpleNamespace(name="STOP"),
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text=text,
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


def _empty_response() -> SimpleNamespace:
    return SimpleNamespace(candidates=[], prompt_feedback=None, usage_metadata=None)


def _make_model(retry_config=None, **kwargs) -> GoogleModel:
    return GoogleModel(
        model="gemini-2.5-pro",
        client=Mock(),
        retry_config=retry_config,
        **kwargs,
    )


def _generate(model: GoogleModel):
    return model.generate_assistant_message([UserMessage("hi")])


# ---- retry_config path: schedule + classification ----


def test_success_on_first_attempt_makes_no_sleep():
    model = _make_model(retry_config=RetryConfig(delays_s=(30.0,)))
    model.client.models.generate_content.side_effect = [_good_response()]
    with patch("time.sleep") as sleep:
        result = _generate(model)
    assert result.content == "hello world"
    assert model.client.models.generate_content.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_retryable_code_retried_then_succeeds(code):
    """Retryable codes sleep delays_s[0] * uniform(0.5, 1.0) — proving the
    legacy 45s minimum-delay floor is not in play."""
    model = _make_model(retry_config=RetryConfig(delays_s=(30.0,)))
    model.client.models.generate_content.side_effect = [
        _api_error(code),
        _good_response(),
    ]
    with patch("time.sleep") as sleep:
        result = _generate(model)
    assert result.content == "hello world"
    assert model.client.models.generate_content.call_count == 2
    sleep_durations = [call.args[0] for call in sleep.call_args_list]
    assert len(sleep_durations) == 1
    assert 15.0 <= sleep_durations[0] <= 30.0


@pytest.mark.parametrize(
    "transport_error",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadError(""),
        ConnectionResetError("connection reset by peer"),
    ],
)
def test_transport_error_retried(transport_error):
    model = _make_model(retry_config=RetryConfig(delays_s=(30.0,)))
    model.client.models.generate_content.side_effect = [
        transport_error,
        _good_response(),
    ]
    with patch("time.sleep"):
        result = _generate(model)
    assert result.content == "hello world"
    assert model.client.models.generate_content.call_count == 2


def test_400_fails_fast_with_native_error():
    model = _make_model(retry_config=RetryConfig(delays_s=(30.0, 60.0)))
    model.client.models.generate_content.side_effect = [_api_error(400)]
    with patch("time.sleep") as sleep:
        with pytest.raises(genai_errors.APIError) as exc_info:
            _generate(model)
    assert exc_info.value.code == 400
    assert not isinstance(exc_info.value, GoogleAPIError)
    assert model.client.models.generate_content.call_count == 1
    sleep.assert_not_called()


def test_exhaustion_reraises_last_native_api_error():
    errors = [_api_error(429), _api_error(429), _api_error(429)]
    model = _make_model(retry_config=RetryConfig(delays_s=(1.0, 2.0)))
    model.client.models.generate_content.side_effect = errors
    with patch("time.sleep") as sleep:
        with pytest.raises(genai_errors.APIError) as exc_info:
            _generate(model)
    assert exc_info.value is errors[-1]
    assert exc_info.value.code == 429
    assert model.client.models.generate_content.call_count == 3
    assert len(sleep.call_args_list) == 2


def test_exhaustion_reraises_transport_error():
    errors = [httpx.ReadError(""), httpx.ReadError("")]
    model = _make_model(retry_config=RetryConfig(delays_s=(1.0,)))
    model.client.models.generate_content.side_effect = errors
    with patch("time.sleep"):
        with pytest.raises(httpx.ReadError) as exc_info:
            _generate(model)
    assert exc_info.value is errors[-1]


def test_deterministic_jitter_gives_exact_sleep_sequence():
    model = _make_model(
        retry_config=RetryConfig(delays_s=(10.0, 20.0, 40.0), jitter=(1.0, 1.0))
    )
    model.client.models.generate_content.side_effect = [
        _api_error(429),
        _api_error(429),
        _api_error(429),
        _good_response(),
    ]
    with patch("time.sleep") as sleep:
        result = _generate(model)
    assert result.content == "hello world"
    sleep_durations = [call.args[0] for call in sleep.call_args_list]
    assert sleep_durations == [10.0, 20.0, 40.0]


def test_processing_errors_are_not_retried_or_rewrapped():
    """Errors from response processing (empty candidates) must surface verbatim
    after a single API call — not classified, retried, or double-wrapped into
    the legacy 'Error generating content: ...' message."""
    model = _make_model(retry_config=RetryConfig(delays_s=(30.0,)))
    model.client.models.generate_content.side_effect = [_empty_response()]
    with patch("time.sleep") as sleep:
        with pytest.raises(GoogleAPIError) as exc_info:
            _generate(model)
    assert str(exc_info.value) == "No content found in Vertex AI response"
    assert model.client.models.generate_content.call_count == 1
    sleep.assert_not_called()


def test_empty_delays_makes_single_native_attempt():
    error = _api_error(429)
    model = _make_model(retry_config=RetryConfig(delays_s=()))
    model.client.models.generate_content.side_effect = [error]
    with patch("time.sleep") as sleep:
        with pytest.raises(genai_errors.APIError) as exc_info:
            _generate(model)
    assert exc_info.value is error
    assert model.client.models.generate_content.call_count == 1
    sleep.assert_not_called()


def test_allow_empty_response_still_returns_empty_message():
    model = _make_model(
        retry_config=RetryConfig(delays_s=(30.0,)), allow_empty_response=True
    )
    model.client.models.generate_content.side_effect = [_empty_response()]
    result = _generate(model)
    assert result.content == ""


# ---- retry_config path: streaming ----


def _stream_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason=None,
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text=text,
                            function_call=None,
                            inline_data=None,
                            thought=False,
                        )
                    ]
                ),
            )
        ],
        usage_metadata=None,
    )


async def _stream_of(chunks):
    for chunk in chunks:
        yield chunk


async def test_streaming_retries_then_yields_chunks():
    model = _make_model(retry_config=RetryConfig(delays_s=(30.0,)))
    model.client.aio.models.generate_content_stream = AsyncMock(
        side_effect=[
            _api_error(429),
            _stream_of([_stream_chunk("hel"), _stream_chunk("lo")]),
        ]
    )
    with patch("patterpunk.lib.retry.asyncio.sleep") as sleep:
        events = [
            event async for event in model.stream_assistant_message([UserMessage("hi")])
        ]
    texts = [event.text for event in events if event.text]
    assert texts == ["hel", "lo"]
    assert model.client.aio.models.generate_content_stream.await_count == 2
    assert len(sleep.call_args_list) == 1
    assert 15.0 <= sleep.call_args_list[0].args[0] <= 30.0


async def test_streaming_exhaustion_reraises_native_error():
    errors = [_api_error(429), _api_error(429)]
    model = _make_model(retry_config=RetryConfig(delays_s=(1.0,)))
    model.client.aio.models.generate_content_stream = AsyncMock(side_effect=errors)
    with patch("patterpunk.lib.retry.asyncio.sleep"):
        with pytest.raises(genai_errors.APIError) as exc_info:
            async for _ in model.stream_assistant_message([UserMessage("hi")]):
                pass
    assert exc_info.value is errors[-1]


# ---- legacy path (retry_config=None): behavior unchanged ----


def test_legacy_429_exhaustion_raises_rate_limit_error():
    model = _make_model()
    model.client.models.generate_content.side_effect = [
        _api_error(429),
        _api_error(429),
        _api_error(429),
    ]
    with patch("time.sleep") as sleep:
        with pytest.raises(GoogleRateLimitError):
            _generate(model)
    assert model.client.models.generate_content.call_count == 3
    sleep_durations = [call.args[0] for call in sleep.call_args_list]
    assert len(sleep_durations) == 2
    assert all(duration >= 45.0 for duration in sleep_durations)


def test_legacy_non_429_api_error_raised_natively():
    model = _make_model()
    model.client.models.generate_content.side_effect = [_api_error(500)]
    with patch("time.sleep"):
        with pytest.raises(genai_errors.APIError) as exc_info:
            _generate(model)
    assert exc_info.value.code == 500
    assert model.client.models.generate_content.call_count == 1


def test_legacy_transport_error_wrapped_in_google_api_error():
    model = _make_model()
    model.client.models.generate_content.side_effect = [
        ConnectionResetError("connection reset by peer")
    ]
    with pytest.raises(GoogleAPIError) as exc_info:
        _generate(model)
    assert str(exc_info.value).startswith("Error generating content:")


# ---- plumbing ----


def test_deepcopy_preserves_retry_config():
    config = RetryConfig(delays_s=(30.0, 900.0))
    model = _make_model(retry_config=config)
    assert deepcopy(model).retry_config == config
