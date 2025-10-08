import pytest
from typing import List, Optional
from unittest.mock import Mock, patch

from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.base import Model
from patterpunk.llm.chunks import CacheChunk, TextChunk, MultimodalChunk
from patterpunk.llm.output_types import OutputType


class MockModel(Model):
    def __init__(self, mock_response="Mock response"):
        self.mock_response = mock_response

    def generate_assistant_message(
        self, messages, tools=None, structured_output=None, output_types=None
    ):
        return AssistantMessage(self.mock_response)

    @staticmethod
    def get_name() -> str:
        return "mock-model"

    @staticmethod
    def get_available_models() -> List[str]:
        return ["mock-model"]


class TestAssistantMessageBackwardCompatibility:
    def test_assistant_message_accepts_string(self):
        message = AssistantMessage("Hello world")
        assert message.content == "Hello world"
        assert message.role == "assistant"

    def test_assistant_message_content_property_returns_string(self):
        message = AssistantMessage("Test content")
        content = message.content
        assert isinstance(content, str)
        assert content == "Test content"


class TestAssistantMessageContentTypeSupport:
    def test_assistant_message_accepts_text_chunk_list(self):
        chunks = [TextChunk("Hello"), TextChunk("world")]
        message = AssistantMessage(chunks)
        assert message.content == "Helloworld"

    def test_assistant_message_accepts_mixed_chunk_list(self):
        chunks = [
            TextChunk("Text part"),
            CacheChunk("Cached content", ttl=300),
            MultimodalChunk.from_file("/app/tests/assets/test.jpg"),
        ]
        message = AssistantMessage(chunks)
        assert "Text part" in message.content

    def test_assistant_message_accepts_cache_chunk_list(self):
        chunks = [CacheChunk("Cached 1", ttl=300), CacheChunk("Cached 2", ttl=600)]
        message = AssistantMessage(chunks)
        assert message.content == "Cached 1Cached 2"

    def test_assistant_message_accepts_multimodal_chunk_list(self):
        chunks = [
            MultimodalChunk.from_file("/app/tests/assets/image.jpg"),
            MultimodalChunk.from_file("/app/tests/assets/photo.png"),
        ]
        message = AssistantMessage(chunks)
        assert isinstance(message.content, str)


class TestContentPropertyBehavior:
    def test_content_property_always_returns_string_for_chunks(self):
        chunks = [TextChunk("Part 1"), TextChunk("Part 2")]
        message = AssistantMessage(chunks)
        content = message.content
        assert isinstance(content, str)
        assert content == "Part 1Part 2"

    def test_content_property_handles_empty_chunk_list(self):
        message = AssistantMessage([])
        assert message.content == ""

    def test_content_property_handles_multimodal_chunks(self):
        chunks = [MultimodalChunk.from_file("/app/tests/assets/test.jpg")]
        message = AssistantMessage(chunks)
        content = message.content
        assert isinstance(content, str)


class TestChunksAccessor:
    def test_chunks_property_returns_list_for_chunk_input(self):
        chunks = [TextChunk("Hello"), TextChunk("world")]
        message = AssistantMessage(chunks)
        assert message.chunks == chunks
        assert len(message.chunks) == 2

    def test_chunks_property_returns_none_for_string_input(self):
        message = AssistantMessage("Hello world")
        assert message.chunks is None

    def test_chunks_property_returns_empty_list_for_empty_input(self):
        message = AssistantMessage([])
        assert message.chunks == []

    def test_chunks_property_preserves_chunk_types(self):
        chunks = [
            TextChunk("Text"),
            CacheChunk("Cache", ttl=300),
            MultimodalChunk.from_file("/app/tests/assets/test.jpg"),
        ]
        message = AssistantMessage(chunks)
        retrieved_chunks = message.chunks
        assert isinstance(retrieved_chunks[0], TextChunk)
        assert isinstance(retrieved_chunks[1], CacheChunk)
        assert isinstance(retrieved_chunks[2], MultimodalChunk)


class TestTypeSpecificAccessors:
    def test_images_property_returns_multimodal_image_chunks(self):
        chunks = [
            TextChunk("Hello"),
            MultimodalChunk.from_file("/app/tests/assets/image.jpg"),
            MultimodalChunk.from_file("/app/tests/assets/video.mp4"),
            MultimodalChunk.from_file("/app/tests/assets/photo.png"),
        ]
        message = AssistantMessage(chunks)
        images = message.images
        assert len(images) == 2
        assert all(chunk.media_type.startswith("image/") for chunk in images)

    def test_videos_property_returns_multimodal_video_chunks(self):
        chunks = [
            TextChunk("Hello"),
            MultimodalChunk.from_file("/app/tests/assets/video.mp4"),
            MultimodalChunk.from_file("/app/tests/assets/movie.avi"),
        ]
        message = AssistantMessage(chunks)
        videos = message.videos
        assert len(videos) == 2
        assert all(chunk.media_type.startswith("video/") for chunk in videos)

    def test_audios_property_returns_multimodal_audio_chunks(self):
        chunks = [
            MultimodalChunk.from_file("/app/tests/assets/audio.mp3"),
            MultimodalChunk.from_file("/app/tests/assets/sound.wav"),
            TextChunk("Text content"),
        ]
        message = AssistantMessage(chunks)
        audios = message.audios
        assert len(audios) == 2
        assert all(chunk.media_type.startswith("audio/") for chunk in audios)

    def test_texts_property_returns_text_and_cache_chunks(self):
        chunks = [
            TextChunk("Text 1"),
            CacheChunk("Cached text", ttl=300),
            MultimodalChunk.from_file("/app/tests/assets/image.jpg"),
            TextChunk("Text 2"),
        ]
        message = AssistantMessage(chunks)
        texts = message.texts
        assert len(texts) == 3
        assert isinstance(texts[0], TextChunk)
        assert isinstance(texts[1], CacheChunk)
        assert isinstance(texts[2], TextChunk)

    def test_type_specific_accessors_return_empty_for_string_content(self):
        message = AssistantMessage("Hello world")
        assert message.images == []
        assert message.videos == []
        assert message.audios == []
        assert message.texts == []

    def test_type_specific_accessors_handle_empty_chunks(self):
        message = AssistantMessage([])
        assert message.images == []
        assert message.videos == []
        assert message.audios == []
        assert message.texts == []


class TestOutputTypeEnum:
    def test_output_type_enum_membership(self):
        assert OutputType.TEXT in OutputType
        assert OutputType.IMAGE in OutputType
        assert OutputType.AUDIO in OutputType
        assert OutputType.VIDEO in OutputType

    def test_output_type_enum_comparison(self):
        assert OutputType.TEXT == OutputType.TEXT
        assert OutputType.TEXT != OutputType.IMAGE


class TestChatCompleteWithOutputTypes:
    def test_complete_accepts_output_types_parameter(self):
        chat = Chat(model=MockModel())
        chat = chat.add_message(UserMessage("Generate an image"))

        result_chat = chat.complete(output_types=[OutputType.IMAGE])
        assert result_chat is not None
        assert len(result_chat.messages) == 2

    def test_complete_accepts_multiple_output_types(self):
        chat = Chat(model=MockModel())
        chat = chat.add_message(UserMessage("Generate content"))

        result_chat = chat.complete(
            output_types=[OutputType.TEXT, OutputType.IMAGE, OutputType.AUDIO]
        )
        assert result_chat is not None

    def test_complete_accepts_empty_output_types_list(self):
        chat = Chat(model=MockModel())
        chat = chat.add_message(UserMessage("Hello"))

        result_chat = chat.complete(output_types=[])
        assert result_chat is not None

    def test_complete_accepts_none_output_types(self):
        chat = Chat(model=MockModel())
        chat = chat.add_message(UserMessage("Hello"))

        result_chat = chat.complete(output_types=None)
        assert result_chat is not None

    def test_complete_passes_output_types_to_model(self):
        with patch.object(MockModel, "generate_assistant_message") as mock_generate:
            mock_generate.return_value = AssistantMessage("Response")

            chat = Chat(model=MockModel())
            chat = chat.add_message(UserMessage("Test"))

            chat.complete(output_types=[OutputType.IMAGE, OutputType.TEXT])

            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert "output_types" in call_args.kwargs
            assert call_args.kwargs["output_types"] == [
                OutputType.IMAGE,
                OutputType.TEXT,
            ]


class TestModelBaseClassSignature:
    def test_model_generate_assistant_message_accepts_output_types(self):
        model = MockModel()
        messages = [UserMessage("Test")]

        result = model.generate_assistant_message(
            messages=messages,
            tools=None,
            structured_output=None,
            output_types=[OutputType.TEXT],
        )

        assert isinstance(result, AssistantMessage)

    def test_model_generate_assistant_message_output_types_optional(self):
        model = MockModel()
        messages = [UserMessage("Test")]

        result = model.generate_assistant_message(messages=messages)
        assert isinstance(result, AssistantMessage)

    def test_model_signature_inspection(self):
        import inspect

        sig = inspect.signature(MockModel.generate_assistant_message)
        assert "output_types" in sig.parameters
        output_types_param = sig.parameters["output_types"]
        assert output_types_param.default is None


class TestMultimodalOutputIntegration:
    def test_assistant_message_with_generated_image(self):
        mock_image_chunk = MultimodalChunk.from_file("/app/tests/assets/generated.jpg")
        chunks = [TextChunk("Here's your image:"), mock_image_chunk]
        message = AssistantMessage(chunks)

        assert len(message.chunks) == 2
        assert len(message.images) == 1
        assert len(message.texts) == 1
        assert "Here's your image:" in message.content

    def test_assistant_message_with_multiple_media_types(self):
        chunks = [
            TextChunk("Generated content:"),
            MultimodalChunk.from_file("/app/tests/assets/image.png"),
            MultimodalChunk.from_file("/app/tests/assets/audio.mp3"),
            MultimodalChunk.from_file("/app/tests/assets/video.mp4"),
        ]
        message = AssistantMessage(chunks)

        assert len(message.images) == 1
        assert len(message.audios) == 1
        assert len(message.videos) == 1
        assert len(message.texts) == 1

    def test_chat_workflow_with_multimodal_output(self):
        mock_chunks = [
            TextChunk("Generated image:"),
            MultimodalChunk.from_file("/app/tests/assets/result.jpg"),
        ]

        with patch.object(MockModel, "generate_assistant_message") as mock_generate:
            mock_generate.return_value = AssistantMessage(mock_chunks)

            mock_model = MockModel()
            chat = Chat(model=mock_model)
            chat = chat.add_message(UserMessage("Create an image"))
            result = chat.complete(output_types=[OutputType.IMAGE])

            assert len(result.latest_message.images) == 1
            assert len(result.latest_message.texts) == 1
            mock_generate.assert_called_once()
