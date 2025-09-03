import pytest
from typing import List, Union

from patterpunk.config.providers.google import GOOGLE_APPLICATION_CREDENTIALS
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage, AssistantMessage
from patterpunk.llm.multimodal import MultimodalChunk
from patterpunk.llm.text import TextChunk
from tests.test_utils import get_resource


def test_google_multimodal_text_and_image_generation():
    if not GOOGLE_APPLICATION_CREDENTIALS:
        pytest.skip("Google credentials not available")

    model = GoogleModel(
        model="gemini-1.5-pro-002", location="us-central1", temperature=0.7
    )

    chat = Chat(model=model)

    response = (
        chat.add_message(
            UserMessage(
                [
                    TextChunk(
                        "Look at this image and generate a detailed description, then create an image of a red apple. Provide both text description and generated image."
                    ),
                    MultimodalChunk.from_file(get_resource("ducks_pond.jpg")),
                ]
            )
        )
        .complete()
        .latest_message
    )

    assert isinstance(response, AssistantMessage)
    assert response.content is not None

    if isinstance(response.content, str):
        assert len(response.content) > 20
        assert "apple" in response.content.lower()
    elif isinstance(response.content, list):
        text_chunks = [
            chunk for chunk in response.content if isinstance(chunk, (str, TextChunk))
        ]
        image_chunks = [
            chunk for chunk in response.content if isinstance(chunk, MultimodalChunk)
        ]

        assert len(text_chunks) > 0
        assert len(image_chunks) > 0

        text_content = ""
        if text_chunks:
            if isinstance(text_chunks[0], str):
                text_content = text_chunks[0]
            else:
                text_content = text_chunks[0].content

        assert len(text_content) > 20
        assert "apple" in text_content.lower()

        for image_chunk in image_chunks:
            assert isinstance(image_chunk, MultimodalChunk)
            assert image_chunk.media_type is not None
            assert image_chunk.media_type.startswith("image/")

            image_data = image_chunk.to_bytes()
            assert len(image_data) > 100
            assert image_data[:4] in [
                b"\xff\xd8\xff\xe0",
                b"\x89PNG",
                b"GIF8",
                b"RIFF",
                b"<svg",
            ]
