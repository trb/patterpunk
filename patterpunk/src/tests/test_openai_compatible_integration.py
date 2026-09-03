import os

import pytest
from pydantic import BaseModel, Field

from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.openai_compatible import OpenAiCompatibleModel

COMPAT_BASE_URL = os.getenv("PP_TEST_OPENAI_COMPAT_BASE_URL")
COMPAT_MODEL = os.getenv("PP_TEST_OPENAI_COMPAT_MODEL")
COMPAT_API_KEY = os.getenv("PP_TEST_OPENAI_COMPAT_API_KEY")

pytestmark = pytest.mark.skipif(
    not (COMPAT_BASE_URL and COMPAT_MODEL),
    reason="No OpenAI-compatible test endpoint configured "
    "(PP_TEST_OPENAI_COMPAT_BASE_URL and PP_TEST_OPENAI_COMPAT_MODEL required).",
)


def make_model(**kwargs):
    return OpenAiCompatibleModel(
        model=COMPAT_MODEL,
        base_url=COMPAT_BASE_URL,
        api_key=COMPAT_API_KEY,
        **kwargs,
    )


def test_basic_completion():
    chat = Chat(model=make_model(temperature=0.1, max_tokens=256))
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

    chat = Chat(model=make_model(max_tokens=512))
    response = chat.add_message(
        UserMessage("What country is Paris in?", structured_output=CityFact)
    ).complete()
    parsed = response.latest_message.parsed_output
    assert parsed.country.lower() == "france"


async def test_streaming():
    model = make_model(temperature=0.1, max_tokens=256)
    collected_text = []
    async for chunk in model.stream_assistant_message(
        [UserMessage("Count from 1 to 5, digits only, comma separated.")]
    ):
        if chunk.text:
            collected_text.append(chunk.text)
    streamed = "".join(collected_text)
    assert "1" in streamed and "5" in streamed
