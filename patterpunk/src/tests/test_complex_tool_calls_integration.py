import pytest

from patterpunk.config.providers.openai import OPENAI_API_KEY
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.models.azure_openai import AzureOpenAiModel
from patterpunk.llm.models.bedrock import BedrockModel
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.models.openai_compatible import OpenAiCompatibleModel
from tests.complex_tool import (
    CONTACT_IDENTIFIER,
    PROMPT,
    assert_tool_received_expected_arguments,
    make_record_contact_tool,
)

pytestmark = pytest.mark.integration

LIVE_MODEL_BUILDERS = {
    "openai": lambda: OpenAiModel(model="gpt-4.1-mini", temperature=0.0),
    "azure_openai": lambda: AzureOpenAiModel(deployment_name="gpt-4", temperature=0.0),
    "openai_compatible": lambda: OpenAiCompatibleModel(
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key=OPENAI_API_KEY,
        temperature=0.0,
    ),
    "anthropic": lambda: AnthropicModel(
        model="claude-haiku-4-5-20251001", max_tokens=1024, temperature=0.0
    ),
    "bedrock_claude": lambda: BedrockModel(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", temperature=0.0
    ),
    "bedrock_nova": lambda: BedrockModel(model_id="us.amazon.nova-2-lite-v1:0"),
    "google": lambda: GoogleModel(
        model="gemini-2.5-flash", location="us-central1", temperature=0.0
    ),
}


def build_chat(provider, record_contact):
    return (
        Chat(model=LIVE_MODEL_BUILDERS[provider]())
        .with_tools([record_contact])
        .add_message(UserMessage(PROMPT))
    )


@pytest.mark.parametrize("provider", sorted(LIVE_MODEL_BUILDERS))
def test_complete_executes_complex_tool_call(provider):
    record_contact, recorded_calls = make_record_contact_tool()

    final_chat = build_chat(provider, record_contact).complete()

    assert_tool_received_expected_arguments(recorded_calls)
    assert CONTACT_IDENTIFIER in final_chat.latest_message.content


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", sorted(LIVE_MODEL_BUILDERS))
async def test_complete_stream_executes_complex_tool_call(provider):
    record_contact, recorded_calls = make_record_contact_tool()

    async with build_chat(provider, record_contact).complete_stream() as stream:
        async for _ in stream.content_delta:
            pass
    final_chat = await stream.chat

    assert_tool_received_expected_arguments(recorded_calls)
    assert CONTACT_IDENTIFIER in final_chat.latest_message.content
