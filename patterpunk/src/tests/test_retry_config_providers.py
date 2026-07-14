"""Wiring tests: every provider model routes through the shared RetryConfig
executor when one is configured.

The schedule/classification/exhaustion semantics themselves are proven with
plain callables in test_retry.py; these tests only prove the ROUTING — that
each provider's retry seam honors retry_config (retry-then-success, native
error on exhaustion) and that the config survives Chat's deepcopy. No live
API calls are made: SDK clients are replaced with Mocks or monkeypatched.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import APIError

from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models import anthropic as anthropic_module
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.models.azure_openai import AzureOpenAiModel
from patterpunk.llm.models.bedrock import BedrockModel
from patterpunk.llm.models.ollama import OllamaModel
from patterpunk.llm.models.openai import OpenAiModel, OpenAiApiError
from patterpunk.llm.retry_config import RetryConfig

TWO_ATTEMPT_CONFIG = RetryConfig(delays_s=(10.0,))


def _transport_error():
    return ConnectionResetError("connection reset by peer")


def _flaky_mock(result):
    """Mock that fails once with a retryable error, then returns result."""
    return Mock(side_effect=[_transport_error(), result])


# ---- OpenAI ----


def _make_openai_model(**kwargs) -> OpenAiModel:
    model = OpenAiModel(
        model="gpt-4.1", _INTERNAL__skip_client_validation=True, **kwargs
    )
    model._client = Mock()
    return model


def test_openai_retries_then_succeeds():
    model = _make_openai_model(retry_config=TWO_ATTEMPT_CONFIG)
    sentinel = SimpleNamespace(output_text="ok")
    model._client.responses.create = _flaky_mock(sentinel)
    with patch("time.sleep") as sleep:
        result = model._execute_with_retry({"model": "gpt-4.1"})
    assert result is sentinel
    assert model._client.responses.create.call_count == 2
    assert len(sleep.call_args_list) == 1


def test_openai_exhaustion_reraises_native_error():
    errors = [_transport_error(), _transport_error()]
    model = _make_openai_model(retry_config=TWO_ATTEMPT_CONFIG)
    model._client.responses.create = Mock(side_effect=errors)
    with patch("time.sleep"):
        with pytest.raises(ConnectionResetError) as exc_info:
            model._execute_with_retry({"model": "gpt-4.1"})
    assert exc_info.value is errors[-1]


def test_openai_legacy_exhaustion_still_wraps():
    """retry_config=None keeps the legacy loop: APIErrors are retried up to
    OPENAI_MAX_RETRIES and exhaustion wraps into OpenAiApiError."""
    model = _make_openai_model()
    api_error = APIError(
        "simulated 500", httpx.Request("POST", "http://test"), body=None
    )
    model._client.responses.create = Mock(side_effect=api_error)
    with patch("time.sleep"):
        with pytest.raises(OpenAiApiError):
            model._execute_with_retry({"model": "gpt-4.1"})


def test_openai_deepcopy_preserves_retry_config():
    model = _make_openai_model(retry_config=TWO_ATTEMPT_CONFIG)
    assert deepcopy(model).retry_config == TWO_ATTEMPT_CONFIG


# ---- Azure OpenAI ----


def _make_azure_model(**kwargs) -> AzureOpenAiModel:
    model = AzureOpenAiModel(deployment_name="gpt-4", **kwargs)
    model._azure_client = Mock()
    return model


def test_azure_retries_then_succeeds():
    model = _make_azure_model(retry_config=TWO_ATTEMPT_CONFIG)
    sentinel = SimpleNamespace(output_text="ok")
    model._azure_client.responses.create = _flaky_mock(sentinel)
    with patch("time.sleep") as sleep:
        result = model._execute_with_retry({"model": "gpt-4"})
    assert result is sentinel
    assert model._azure_client.responses.create.call_count == 2
    assert len(sleep.call_args_list) == 1


def test_azure_exhaustion_reraises_native_error():
    errors = [_transport_error(), _transport_error()]
    model = _make_azure_model(retry_config=TWO_ATTEMPT_CONFIG)
    model._azure_client.responses.create = Mock(side_effect=errors)
    with patch("time.sleep"):
        with pytest.raises(ConnectionResetError) as exc_info:
            model._execute_with_retry({"model": "gpt-4"})
    assert exc_info.value is errors[-1]


def test_azure_deepcopy_preserves_retry_config():
    model = _make_azure_model(retry_config=TWO_ATTEMPT_CONFIG)
    assert deepcopy(model).retry_config == TWO_ATTEMPT_CONFIG


# ---- Anthropic ----


def _fake_anthropic_response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=text)],
        id="msg_test",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def _fake_anthropic_singleton(create_mock) -> SimpleNamespace:
    """Fake for the module-level anthropic singleton: exposes messages.create
    (legacy path) and with_options(max_retries=0) (retry_config path), both
    routing to the same create mock."""
    client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))

    def with_options(max_retries):
        assert max_retries == 0
        return client

    return SimpleNamespace(
        messages=client.messages,
        with_options=with_options,
    )


def test_anthropic_retries_then_succeeds(monkeypatch):
    create = _flaky_mock(_fake_anthropic_response("hello"))
    monkeypatch.setattr(
        anthropic_module, "anthropic", _fake_anthropic_singleton(create)
    )
    model = AnthropicModel(model="claude-sonnet-4-5", retry_config=TWO_ATTEMPT_CONFIG)
    with patch("time.sleep") as sleep:
        result = model._execute_with_retry_loop([UserMessage("hi")], None, None)
    assert result.content == "hello"
    assert create.call_count == 2
    assert len(sleep.call_args_list) == 1


def test_anthropic_exhaustion_reraises_native_error(monkeypatch):
    errors = [_transport_error(), _transport_error()]
    create = Mock(side_effect=errors)
    monkeypatch.setattr(
        anthropic_module, "anthropic", _fake_anthropic_singleton(create)
    )
    model = AnthropicModel(model="claude-sonnet-4-5", retry_config=TWO_ATTEMPT_CONFIG)
    with patch("time.sleep"):
        with pytest.raises(ConnectionResetError) as exc_info:
            model._execute_with_retry_loop([UserMessage("hi")], None, None)
    assert exc_info.value is errors[-1]


def test_anthropic_deepcopy_preserves_retry_config():
    model = AnthropicModel(model="claude-sonnet-4-5", retry_config=TWO_ATTEMPT_CONFIG)
    assert deepcopy(model).retry_config == TWO_ATTEMPT_CONFIG


# ---- Bedrock ----


def _make_bedrock_model(**kwargs) -> BedrockModel:
    return BedrockModel(
        model_id="anthropic.claude-3", region_name="us-east-1", **kwargs
    )


def test_bedrock_retries_then_succeeds():
    model = _make_bedrock_model(retry_config=TWO_ATTEMPT_CONFIG)
    operation = _flaky_mock({"output": "ok"})
    with patch("time.sleep") as sleep:
        result = model._execute_with_retry(operation, "test operation")
    assert result == {"output": "ok"}
    assert operation.call_count == 2
    assert len(sleep.call_args_list) == 1


def test_bedrock_exhaustion_reraises_native_error():
    errors = [_transport_error(), _transport_error()]
    model = _make_bedrock_model(retry_config=TWO_ATTEMPT_CONFIG)
    operation = Mock(side_effect=errors)
    with patch("time.sleep"):
        with pytest.raises(ConnectionResetError) as exc_info:
            model._execute_with_retry(operation, "test operation")
    assert exc_info.value is errors[-1]


def test_bedrock_disables_sdk_retries_when_retry_config_set():
    """botocore's own retry layer must not stack attempts under the schedule."""
    model = _make_bedrock_model(retry_config=TWO_ATTEMPT_CONFIG)
    assert model.client.meta.config.retries["total_max_attempts"] == 1
    legacy_model = _make_bedrock_model()
    legacy_retries = legacy_model.client.meta.config.retries or {}
    assert legacy_retries.get("total_max_attempts") != 1


def test_bedrock_deepcopy_preserves_retry_config():
    model = _make_bedrock_model(retry_config=TWO_ATTEMPT_CONFIG)
    assert deepcopy(model).retry_config == TWO_ATTEMPT_CONFIG


# ---- Ollama ----


def _make_ollama_model(**kwargs) -> OllamaModel:
    model = OllamaModel(model="llama3", **kwargs)
    model._client = Mock()
    return model


def test_ollama_retries_then_succeeds():
    model = _make_ollama_model(retry_config=TWO_ATTEMPT_CONFIG)
    model._client.chat = _flaky_mock({"message": {"content": "ok"}})
    with patch("time.sleep") as sleep:
        result = model._execute_with_retry({"model": "llama3", "messages": []})
    assert result == {"message": {"content": "ok"}}
    assert model._client.chat.call_count == 2
    assert len(sleep.call_args_list) == 1


def test_ollama_exhaustion_reraises_native_error():
    errors = [_transport_error(), _transport_error()]
    model = _make_ollama_model(retry_config=TWO_ATTEMPT_CONFIG)
    model._client.chat = Mock(side_effect=errors)
    with patch("time.sleep"):
        with pytest.raises(ConnectionResetError) as exc_info:
            model._execute_with_retry({"model": "llama3", "messages": []})
    assert exc_info.value is errors[-1]


def test_ollama_deepcopy_preserves_retry_config():
    model = _make_ollama_model(retry_config=TWO_ATTEMPT_CONFIG)
    assert deepcopy(model).retry_config == TWO_ATTEMPT_CONFIG
