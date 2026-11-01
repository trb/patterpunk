from unittest.mock import Mock

import pytest

from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.models.anthropic_foundry import AnthropicFoundryModel
from patterpunk.llm.models.anthropic_vertex import AnthropicVertexModel
from patterpunk.llm.models.azure_openai import AzureOpenAiModel
from patterpunk.llm.models.bedrock import BedrockModel
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.models.openai_compatible import OpenAiCompatibleModel
from tests.complex_tool import PROMPT, record_contact_tool_definition

TOOLS = [record_contact_tool_definition()]
PARAMETER_SCHEMA = TOOLS[0]["function"]["parameters"]
MESSAGES = [SystemMessage("You record contacts."), UserMessage(PROMPT)]


def openai_sent_schema():
    model = OpenAiModel(model="gpt-4.1-mini")
    request = model._prepare_request_parameters(MESSAGES, TOOLS, None, None)
    return request["tools"][0]["parameters"]


def openai_compatible_sent_schema():
    model = OpenAiCompatibleModel(
        model="gpt-4.1-mini", base_url="http://localhost:8000/v1", api_key="unused"
    )
    request = model._build_request_params(MESSAGES, TOOLS, None)
    return request["tools"][0]["function"]["parameters"]


def anthropic_sent_schema():
    model = AnthropicModel(model="claude-haiku-4-5-20251001")
    request = model._configure_tools_and_structured_output({}, TOOLS, None)
    return request["tools"][0]["input_schema"]


def bedrock_sent_schema():
    model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    request = model._build_converse_params(MESSAGES, TOOLS)
    return request["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]


def google_sent_schema():
    model = GoogleModel(model="gemini-2.5-flash", client=Mock())
    _, config, _ = model._prepare_generation_request(MESSAGES, TOOLS)
    return config.tools[0].function_declarations[0].parameters_json_schema


SENT_SCHEMA_BY_PROVIDER = {
    "openai": openai_sent_schema,
    "openai_compatible": openai_compatible_sent_schema,
    "anthropic": anthropic_sent_schema,
    "bedrock": bedrock_sent_schema,
    "google": google_sent_schema,
}


def test_fixture_schema_contains_every_complex_feature():
    properties = PARAMETER_SCHEMA["properties"]
    assert properties["names"]["items"] == {"type": "string"}
    assert properties["scores"]["items"] == {"type": "integer"}
    assert properties["address"]["$ref"] == "#/$defs/Address"
    assert properties["previous_addresses"]["items"] == {"$ref": "#/$defs/Address"}
    assert properties["priority"]["$ref"] == "#/$defs/Priority"
    assert {"type": "null"} in properties["note"]["anyOf"]
    assert PARAMETER_SCHEMA["$defs"]["Address"]["required"] == ["street", "city"]
    assert PARAMETER_SCHEMA["$defs"]["Priority"]["enum"] == ["low", "high"]
    assert "note" not in PARAMETER_SCHEMA["required"]


@pytest.mark.parametrize("provider", sorted(SENT_SCHEMA_BY_PROVIDER))
def test_request_carries_the_complete_parameter_schema(provider):
    assert SENT_SCHEMA_BY_PROVIDER[provider]() == PARAMETER_SCHEMA


@pytest.mark.parametrize(
    "variant, base_class, method_name",
    [
        (AzureOpenAiModel, OpenAiModel, "_prepare_request_parameters"),
        (AzureOpenAiModel, OpenAiModel, "_convert_tools_to_responses_format"),
        (
            AnthropicVertexModel,
            AnthropicModel,
            "_configure_tools_and_structured_output",
        ),
        (AnthropicVertexModel, AnthropicModel, "_convert_tools_to_anthropic_format"),
        (
            AnthropicFoundryModel,
            AnthropicModel,
            "_configure_tools_and_structured_output",
        ),
        (AnthropicFoundryModel, AnthropicModel, "_convert_tools_to_anthropic_format"),
    ],
)
def test_provider_variant_builds_tool_requests_like_its_base_class(
    variant, base_class, method_name
):
    assert getattr(variant, method_name) is getattr(base_class, method_name)
