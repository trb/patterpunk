import copy
import importlib
from unittest.mock import Mock

import pytest

import patterpunk.llm.models.bedrock as bedrock_module
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.bedrock import BedrockModel


class BedrockModelWithRequestFields(BedrockModel):
    def __init__(self, *args, top_k=None, stop_sequences=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.stop_sequences = stop_sequences

    def _build_converse_params(self, messages, tools=None, structured_output=None):
        params = super()._build_converse_params(messages, tools, structured_output)
        if self.stop_sequences:
            params.setdefault("inferenceConfig", {})[
                "stopSequences"
            ] = self.stop_sequences
        if self.top_k is not None:
            request_fields = params.setdefault("additionalModelRequestFields", {})
            request_fields.setdefault("inferenceConfig", {})["topK"] = self.top_k
        return params


@pytest.fixture
def shared_bedrock_client(monkeypatch):
    client = Mock()
    client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
    }
    monkeypatch.setattr(
        bedrock_module, "get_bedrock_client_by_region", lambda **_: client
    )
    return client


def build_model_with_request_fields():
    return BedrockModelWithRequestFields(
        model_id="ca.amazon.nova-lite-v1:0",
        temperature=0.0,
        max_tokens=3000,
        region_name="ca-central-1",
        top_k=1,
        stop_sequences=["</summary>"],
    )


def test_deepcopy_keeps_the_subclass_and_its_attributes(shared_bedrock_client):
    model = build_model_with_request_fields()

    copied_model = copy.deepcopy(model)

    assert type(copied_model) is BedrockModelWithRequestFields
    assert copied_model is not model
    assert copied_model.top_k == 1
    assert copied_model.stop_sequences == ["</summary>"]
    assert copied_model.max_tokens == 3000


def test_deepcopy_copies_mutable_settings_and_shares_the_client(shared_bedrock_client):
    model = build_model_with_request_fields()

    copied_model = copy.deepcopy(model)

    assert copied_model.stop_sequences is not model.stop_sequences
    assert copied_model.client is model.client


def test_chat_sends_the_request_fields_of_a_model_subclass(shared_bedrock_client):
    model = build_model_with_request_fields()

    (
        Chat(model=model)
        .add_message(SystemMessage("system"))
        .add_message(UserMessage("user"))
        .complete()
    )

    sent_request = shared_bedrock_client.converse.call_args.kwargs
    assert sent_request["additionalModelRequestFields"] == {
        "inferenceConfig": {"topK": 1}
    }
    assert sent_request["inferenceConfig"]["stopSequences"] == ["</summary>"]
    assert sent_request["inferenceConfig"]["maxTokens"] == 3000


MODEL_SHARED_ATTRIBUTES = [
    ("patterpunk.llm.models.bedrock", "BedrockModel", ["client"]),
    ("patterpunk.llm.models.google", "GoogleModel", ["client"]),
    ("patterpunk.llm.models.openai", "OpenAiModel", ["_client", "_async_client"]),
    (
        "patterpunk.llm.models.azure_openai",
        "AzureOpenAiModel",
        [
            "_azure_client",
            "_azure_async_client",
            "_azure_reasoning_client",
            "_azure_reasoning_async_client",
        ],
    ),
    ("patterpunk.llm.models.ollama", "OllamaModel", ["_client"]),
    (
        "patterpunk.llm.models.openai_compatible",
        "OpenAiCompatibleModel",
        ["_client", "_async_client", "api_key_provider"],
    ),
    (
        "patterpunk.llm.models.anthropic_foundry",
        "AnthropicFoundryModel",
        ["azure_ad_token_provider"],
    ),
]


def build_uninitialized_subclass_instance(module_name, class_name):
    model_class = getattr(importlib.import_module(module_name), class_name)
    subclass = type(f"{class_name}WithSettings", (model_class,), {})
    return subclass.__new__(subclass)


@pytest.mark.parametrize(
    "module_name, class_name, shared_attribute_names", MODEL_SHARED_ATTRIBUTES
)
def test_every_model_copy_keeps_its_subclass_and_shares_its_clients(
    module_name, class_name, shared_attribute_names
):
    model = build_uninitialized_subclass_instance(module_name, class_name)
    for attribute_name in shared_attribute_names:
        setattr(model, attribute_name, object())
    model.extra_settings = ["value"]

    copied_model = copy.deepcopy(model)

    assert type(copied_model) is type(model)
    assert copied_model.extra_settings == ["value"]
    assert copied_model.extra_settings is not model.extra_settings
    for attribute_name in shared_attribute_names:
        assert getattr(copied_model, attribute_name) is getattr(model, attribute_name)


def test_ollama_copy_starts_without_temp_files():
    model = build_uninitialized_subclass_instance(
        "patterpunk.llm.models.ollama", "OllamaModel"
    )
    model._client = object()
    model._temp_files = ["/tmp/patterpunk-image.png"]

    copied_model = copy.deepcopy(model)

    assert not hasattr(copied_model, "_temp_files")
