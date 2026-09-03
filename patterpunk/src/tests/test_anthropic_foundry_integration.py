import pytest

from patterpunk.config.providers.anthropic_foundry import is_anthropic_foundry_available
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.anthropic_foundry import AnthropicFoundryModel
from patterpunk.llm.models.base import TokenCountingError

pytestmark = pytest.mark.skipif(
    not is_anthropic_foundry_available(),
    reason="Anthropic on Microsoft Foundry is not configured "
    "(PP_ANTHROPIC_FOUNDRY_RESOURCE/BASE_URL and PP_ANTHROPIC_FOUNDRY_API_KEY required).",
)

DEPLOYMENT = "claude-sonnet-4-5"


def test_basic_completion():
    chat = Chat(
        model=AnthropicFoundryModel(
            deployment_name=DEPLOYMENT, max_tokens=1024, temperature=0.1
        )
    )
    response = (
        chat.add_message(SystemMessage("Answer with just the number."))
        .add_message(UserMessage("What is 2 + 2?"))
        .complete()
    )
    assert "4" in response.latest_message.content


def test_count_tokens_support():
    """
    Microsoft's Foundry documentation is silent on the count_tokens endpoint.
    Until that is settled, this test cannot assert a single outcome.
    """
    model = AnthropicFoundryModel(deployment_name=DEPLOYMENT)
    try:
        token_count = model.count_tokens("The quick brown fox jumps over the lazy dog.")
    except TokenCountingError as error:
        print(f"Foundry does not serve count_tokens: {error}")
    else:
        print(f"Foundry serves count_tokens: {token_count} tokens")
        assert token_count > 5
