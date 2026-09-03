import pytest
from pydantic import BaseModel, Field

from patterpunk.config.providers.anthropic_vertex import is_anthropic_vertex_available
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.anthropic_vertex import AnthropicVertexModel
from patterpunk.llm.thinking import ThinkingConfig

pytestmark = pytest.mark.skipif(
    not is_anthropic_vertex_available(),
    reason="Anthropic on Vertex is not configured "
    "(PP_GOOGLE_APPLICATION_CREDENTIALS with a resolvable project required).",
)


def test_basic_completion():
    chat = Chat(
        model=AnthropicVertexModel(
            model="claude-sonnet-4-5@20250929", max_tokens=1024, temperature=0.1
        )
    )
    response = (
        chat.add_message(SystemMessage("Answer with just the number."))
        .add_message(UserMessage("What is 2 + 2?"))
        .complete()
    )
    assert "4" in response.latest_message.content


def test_structured_output():
    class CityFact(BaseModel):
        city: str = Field(description="The city asked about")
        country: str = Field(description="The country the city is in")

    chat = Chat(
        model=AnthropicVertexModel(model="claude-sonnet-4-5@20250929", max_tokens=1024)
    )
    response = chat.add_message(
        UserMessage("What country is Paris in?", structured_output=CityFact)
    ).complete()
    parsed = response.latest_message.parsed_output
    assert parsed.country.lower() == "france"


def test_thinking_completion():
    chat = Chat(
        model=AnthropicVertexModel(
            model="claude-sonnet-4-5@20250929",
            max_tokens=8000,
            thinking_config=ThinkingConfig(token_budget=2000),
        )
    )
    response = chat.add_message(
        UserMessage("Why is the sky blue? One sentence.")
    ).complete()
    assert len(response.latest_message.content.strip()) > 0


def test_count_tokens():
    model = AnthropicVertexModel(model="claude-sonnet-4-5@20250929")
    token_count = model.count_tokens("The quick brown fox jumps over the lazy dog.")
    assert token_count > 5
