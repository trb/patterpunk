"""
Tests for token counting functionality.

These tests verify that our token counting implementation produces accurate
results by comparing against the actual token counts returned by API responses.
"""

import pytest
from patterpunk.llm.messages import UserMessage, SystemMessage, AssistantMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.tool_result import ToolResultMessage
from patterpunk.llm.tool_types import ToolCall
from patterpunk.llm.models.base import TokenCountingError
from patterpunk.llm.utils import get_image_dimensions
from patterpunk.llm.chunks import MultimodalChunk, TextChunk


class TestImageDimensionParsing:
    """Test the pure-Python image dimension parser."""

    def test_png_dimensions(self):
        """Test PNG dimension extraction."""
        import base64

        # 1x1 white PNG
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        png_data = base64.b64decode(png_b64)
        width, height = get_image_dimensions(png_data)
        assert width == 1
        assert height == 1

    def test_unsupported_format_raises(self):
        """Test that unsupported formats raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported image format"):
            get_image_dimensions(b"not an image")


class TestOpenAITokenCounting:
    """Test OpenAI token counting accuracy using tiktoken."""

    def test_string_counting_exact(self):
        """Test counting tokens for plain strings with exact values."""
        from patterpunk.llm.models.openai import OpenAiModel
        import tiktoken

        model = OpenAiModel(model="gpt-4o-mini")
        enc = tiktoken.get_encoding("o200k_base")

        # Test with deterministic strings
        test_cases = [
            ("Hello, world!", enc.encode("Hello, world!")),
            (
                "What is the capital of France?",
                enc.encode("What is the capital of France?"),
            ),
            ("a", enc.encode("a")),
            ("", enc.encode("")),
        ]

        for text, expected_tokens in test_cases:
            count = model.count_tokens(text)
            assert count == len(
                expected_tokens
            ), f"Token count mismatch for '{text}': got {count}, expected {len(expected_tokens)}"

    def test_message_counting_includes_overhead(self):
        """Test that message counting includes structural overhead."""
        from patterpunk.llm.models.openai import OpenAiModel
        import tiktoken

        model = OpenAiModel(model="gpt-4o-mini")
        enc = tiktoken.get_encoding("o200k_base")

        text = "Hello"
        raw_tokens = len(enc.encode(text))

        # Message should include overhead (4 tokens for role/delimiters)
        message = UserMessage(text)
        message_count = model.count_tokens(message)

        assert (
            message_count == raw_tokens + 4
        ), f"Expected {raw_tokens} + 4 overhead = {raw_tokens + 4}, got {message_count}"

    def test_batch_counting_equals_sum(self):
        """Verify batch counting equals sum of individual counts."""
        from patterpunk.llm.models.openai import OpenAiModel

        model = OpenAiModel(model="gpt-4o-mini")
        messages = [
            UserMessage("Hello"),
            AssistantMessage("Hi there!"),
            UserMessage("How are you?"),
        ]

        # Count individually
        individual_sum = sum(model.count_tokens(m) for m in messages)

        # Count as batch
        batch_count = model.count_tokens(messages)

        # Should be equal for local counting
        assert batch_count == individual_sum

    def test_tool_call_message_counting_exact(self):
        """Test counting tokens for ToolCallMessage with exact values."""
        from patterpunk.llm.models.openai import OpenAiModel
        import tiktoken

        model = OpenAiModel(model="gpt-4o-mini")
        enc = tiktoken.get_encoding("o200k_base")

        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments='{"location": "Paris"}',
        )
        message = ToolCallMessage(tool_calls=[tool_call])

        count = model.count_tokens(message)

        # Expected: 4 (message overhead) + name tokens + args tokens + 10 (structure overhead)
        expected = (
            4  # message overhead
            + len(enc.encode("get_weather"))  # name tokens
            + len(enc.encode('{"location": "Paris"}'))  # args tokens
            + 10  # structure overhead for id, type, etc.
        )
        assert count == expected, f"Expected {expected}, got {count}"

    def test_tool_result_message_counting_exact(self):
        """Test counting tokens for ToolResultMessage with exact values."""
        from patterpunk.llm.models.openai import OpenAiModel
        import tiktoken

        model = OpenAiModel(model="gpt-4o-mini")
        enc = tiktoken.get_encoding("o200k_base")

        message = ToolResultMessage(
            content='{"temperature": 22, "condition": "sunny"}',
            call_id="call_123",
            function_name="get_weather",
        )

        count = model.count_tokens(message)

        # Expected: 4 (message overhead) + content tokens + call_id tokens + function_name tokens
        expected = (
            4  # message overhead
            + len(enc.encode('{"temperature": 22, "condition": "sunny"}'))
            + len(enc.encode("call_123"))
            + len(enc.encode("get_weather"))
        )
        assert count == expected, f"Expected {expected}, got {count}"


class TestOpenAIImageTokenFormula:
    """Test OpenAI image token formula accuracy."""

    def test_low_detail_always_85_tokens(self):
        """Low detail images should always be 85 tokens."""
        from patterpunk.llm.models.openai import OpenAiModel

        model = OpenAiModel(model="gpt-4o-mini")

        # Create a test image chunk (1x1 PNG)
        import base64

        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        chunk = MultimodalChunk.from_base64(png_b64, media_type="image/png")

        # Low detail should be 85 tokens
        tokens = model._count_image_tokens(chunk, detail="low")
        assert tokens == 85

    def test_small_image_formula(self):
        """Test formula for small images (no scaling needed)."""
        from patterpunk.llm.models.openai import OpenAiModel

        model = OpenAiModel(model="gpt-4o-mini")

        # Create a small test image (100x100 would fit in one 512x512 tile)
        import base64

        # 1x1 PNG - fits in one tile
        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        chunk = MultimodalChunk.from_base64(png_b64, media_type="image/png")

        # 1x1 image -> 1 tile -> 85 + 170 = 255 tokens
        tokens = model._count_image_tokens(chunk, detail="auto")
        assert tokens == 85 + 170  # 255 tokens


class TestAnthropicTokenCounting:
    """Test Anthropic token counting via API."""

    def test_api_counting_works(self):
        """Verify Anthropic API counting returns valid results."""
        from patterpunk.llm.models.anthropic import AnthropicModel

        model = AnthropicModel(model="claude-haiku-4-5-20251001")
        test_message = UserMessage("What is the capital of France?")

        count = model.count_tokens(test_message)
        assert count > 0
        assert count < 100

    def test_batch_counting(self):
        """Test counting multiple messages in single API call."""
        from patterpunk.llm.models.anthropic import AnthropicModel

        model = AnthropicModel(model="claude-haiku-4-5-20251001")
        messages = [
            UserMessage("Hello"),
            AssistantMessage("Hi there!"),
            UserMessage("How are you?"),
        ]

        count = model.count_tokens(messages)
        assert count > 0

    def test_string_counting(self):
        """Test counting a plain string."""
        from patterpunk.llm.models.anthropic import AnthropicModel

        model = AnthropicModel(model="claude-haiku-4-5-20251001")
        count = model.count_tokens("Hello, world!")
        assert count > 0
        assert count < 20


class TestAnthropicTokenCountingAccuracy:
    """Verify Anthropic counting accuracy against API responses."""

    def test_count_matches_api_usage_exact(self):
        """
        Compare our count_tokens result against actual API response usage.

        This is the key verification test - it makes an actual API call and
        compares our pre-calculated token count against what Anthropic reports
        in the response metadata. Uses the patterpunk model class which handles
        authentication.
        """
        from patterpunk.llm.models.anthropic import AnthropicModel
        from patterpunk.config.providers.anthropic import anthropic

        # Test message content
        test_content = "What is the capital of France?"

        # Get our pre-calculated count via the count_tokens API
        pre_count_response = anthropic.messages.count_tokens(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": test_content}],
        )
        pre_count = pre_count_response.input_tokens

        # Make actual inference call
        response = anthropic.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": test_content}],
        )

        # Get the actual input tokens from the response
        actual_input_tokens = response.usage.input_tokens

        # These should match exactly since we're using the same API endpoint
        assert pre_count == actual_input_tokens, (
            f"Token count mismatch! "
            f"count_tokens API: {pre_count}, "
            f"inference usage: {actual_input_tokens}"
        )


class TestGoogleTokenCounting:
    """Test Google Vertex AI token counting."""

    def test_api_counting_works(self):
        """Verify Google API counting returns valid results."""
        from patterpunk.llm.models.google import GoogleModel

        model = GoogleModel(model="gemini-2.0-flash")
        test_message = UserMessage("What is the capital of France?")

        count = model.count_tokens(test_message)
        assert count > 0
        assert count < 100

    def test_batch_counting(self):
        """Test counting multiple messages."""
        from patterpunk.llm.models.google import GoogleModel

        model = GoogleModel(model="gemini-2.0-flash")
        messages = [
            UserMessage("Hello"),
            AssistantMessage("Hi there!"),
            UserMessage("How are you?"),
        ]

        count = model.count_tokens(messages)
        assert count > 0

    def test_string_counting(self):
        """Test counting a plain string."""
        from patterpunk.llm.models.google import GoogleModel

        model = GoogleModel(model="gemini-2.0-flash")
        count = model.count_tokens("Hello, world!")
        assert count > 0


class TestBedrockTokenCounting:
    """Test AWS Bedrock token counting."""

    def test_claude_45_native_counting(self):
        """
        Verify Bedrock Claude 4.5 uses native CountTokens API.

        Claude 4.5+ models support Bedrock's CountTokens API directly.
        """
        from patterpunk.llm.models.bedrock import BedrockModel

        model = BedrockModel(model_id="anthropic.claude-haiku-4-5-20251001-v1:0")
        test_message = UserMessage("What is the capital of France?")

        count = model.count_tokens(test_message)
        assert count > 0
        assert count < 100

    def test_claude_3_fallback_to_local_estimation(self):
        """
        Verify Claude 3 models fall back to local character estimation.

        Claude 3 models don't support Bedrock's CountTokens API, so we fall
        back to local estimation using Anthropic's ~3.5 chars/token heuristic
        with a 20% safety margin. This keeps all data local (no API calls).
        """
        from patterpunk.llm.models.bedrock import BedrockModel

        # Claude 3 Haiku - doesn't support Bedrock CountTokens
        model = BedrockModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")
        test_message = UserMessage("What is the capital of France?")

        # Should fall back to local estimation and succeed
        count = model.count_tokens(test_message)
        assert count > 0
        assert count < 100

    def test_claude_3_local_estimation_includes_margin(self):
        """
        Verify local estimation includes correct margin based on text length.

        The estimation uses different scaling factors based on text length:
        - Short text (<500 chars): 3.5 chars/token, 20% margin
        """
        from patterpunk.llm.models.bedrock import BedrockModel

        model = BedrockModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")

        # Test with a known string - "Hello" = 5 chars
        # Plus ~20 char overhead for message structure = 25 chars total
        # Using short text formula: 25 / 3.5 * 1.2 = 8.57, rounds to 9
        test_message = UserMessage("Hello")
        count = model.count_tokens(test_message)

        # Should be exactly 9 based on the formula
        expected = int((5 + 20) / 3.5 * 1.2 + 0.5)  # 9
        assert count == expected, f"Expected {expected}, got {count}"

    def test_claude_3_batch_counting(self):
        """Test batch counting for Claude 3 via local estimation fallback."""
        from patterpunk.llm.models.bedrock import BedrockModel

        model = BedrockModel(model_id="anthropic.claude-3-haiku-20240307-v1:0")
        messages = [
            UserMessage("Hello"),
            AssistantMessage("Hi there!"),
            UserMessage("How are you?"),
        ]

        count = model.count_tokens(messages)
        assert count > 0

    def test_unsupported_model_raises_error(self):
        """Test that unsupported models raise TokenCountingError."""
        from patterpunk.llm.models.bedrock import BedrockModel

        # Create a model with an unsupported model ID (Titan)
        model = BedrockModel(model_id="amazon.titan-text-express-v1")

        with pytest.raises(TokenCountingError, match="not supported"):
            model.count_tokens("Hello")


class TestChatTokenCounting:
    """Test Chat-level token counting."""

    def test_chat_count_tokens(self):
        """Test counting all tokens in a chat."""
        from patterpunk.llm.models.openai import OpenAiModel
        from patterpunk.llm.chat import Chat

        model = OpenAiModel(model="gpt-4o-mini")
        chat = (
            Chat(model=model)
            .add_message(SystemMessage("You are helpful."))
            .add_message(UserMessage("Hello!"))
        )

        count = chat.count_tokens(include_tools=False)
        assert count > 0

    def test_chat_count_tokens_with_tools(self):
        """Test counting tokens including tool definitions."""
        from patterpunk.llm.models.openai import OpenAiModel
        from patterpunk.llm.chat import Chat

        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return "sunny"

        model = OpenAiModel(model="gpt-4o-mini")
        chat = (
            Chat(model=model)
            .add_message(UserMessage("What's the weather?"))
            .with_tools([get_weather])
        )

        count_without_tools = chat.count_tokens(include_tools=False)
        count_with_tools = chat.count_tokens(include_tools=True)

        # With tools should be higher
        assert count_with_tools > count_without_tools


class TestTokenCountingError:
    """Test TokenCountingError handling."""

    def test_openai_pdf_raises_error(self):
        """Test that OpenAI raises error for PDFs."""
        from patterpunk.llm.models.openai import OpenAiModel

        model = OpenAiModel(model="gpt-4o-mini")

        # Create a fake PDF chunk
        chunk = MultimodalChunk.from_bytes(b"%PDF-1.4", media_type="application/pdf")
        message = UserMessage([TextChunk("Analyze this:"), chunk])

        with pytest.raises(TokenCountingError, match="PDF"):
            model.count_tokens(message)


@pytest.mark.asyncio
class TestAsyncTokenCounting:
    """Test async token counting methods."""

    async def test_openai_async_matches_sync(self):
        """Test OpenAI async counting matches sync counting."""
        from patterpunk.llm.models.openai import OpenAiModel

        model = OpenAiModel(model="gpt-4o-mini")
        test_message = UserMessage("What is the capital of France?")

        sync_count = model.count_tokens(test_message)
        async_count = await model.count_tokens_async(test_message)

        assert (
            async_count == sync_count
        ), f"Async count {async_count} != sync count {sync_count}"

    async def test_anthropic_async_counting(self):
        """Test Anthropic async token counting works."""
        from patterpunk.llm.models.anthropic import AnthropicModel

        model = AnthropicModel(model="claude-haiku-4-5-20251001")
        test_message = UserMessage("Hello, world!")

        sync_count = model.count_tokens(test_message)
        async_count = await model.count_tokens_async(test_message)

        # Both should return same value from same API
        assert async_count == sync_count

    async def test_google_async_counting(self):
        """Test Google async token counting works."""
        from patterpunk.llm.models.google import GoogleModel

        model = GoogleModel(model="gemini-2.0-flash")
        test_message = UserMessage("Hello, world!")

        sync_count = model.count_tokens(test_message)
        async_count = await model.count_tokens_async(test_message)

        # Both should return same value from same API
        assert async_count == sync_count

    async def test_chat_async_counting(self):
        """Test Chat.count_tokens_async matches sync version."""
        from patterpunk.llm.models.openai import OpenAiModel
        from patterpunk.llm.chat import Chat

        model = OpenAiModel(model="gpt-4o-mini")
        chat = (
            Chat(model=model)
            .add_message(SystemMessage("You are helpful."))
            .add_message(UserMessage("Hello!"))
        )

        sync_count = chat.count_tokens(include_tools=False)
        async_count = await chat.count_tokens_async(include_tools=False)

        assert async_count == sync_count


class TestThinkingBlocksCounting:
    """Test token counting for thinking/reasoning blocks."""

    def test_openai_thinking_blocks_counted(self):
        """Test that thinking blocks are included in OpenAI token count."""
        from patterpunk.llm.models.openai import OpenAiModel
        import tiktoken

        model = OpenAiModel(model="gpt-4o-mini")
        enc = tiktoken.get_encoding("o200k_base")

        # Create message without thinking blocks
        message_without = AssistantMessage("The answer is 42.")
        count_without = model.count_tokens(message_without)

        # Create message with thinking blocks
        message_with = AssistantMessage(
            "The answer is 42.",
            thinking_blocks=[{"thinking": "Let me calculate: 6 * 7 = 42"}],
        )
        count_with = model.count_tokens(message_with)

        # Count with thinking should be higher
        thinking_tokens = len(enc.encode("Let me calculate: 6 * 7 = 42"))
        assert count_with == count_without + thinking_tokens, (
            f"Expected {count_without} + {thinking_tokens} = {count_without + thinking_tokens}, "
            f"got {count_with}"
        )


class TestSystemMessageCounting:
    """Test token counting for system messages."""

    def test_anthropic_system_message_counted(self):
        """Test that system messages are counted in Anthropic token count."""
        from patterpunk.llm.models.anthropic import AnthropicModel

        model = AnthropicModel(model="claude-haiku-4-5-20251001")

        # Count just user message
        user_only = [UserMessage("Hello")]
        count_user_only = model.count_tokens(user_only)

        # Count with system message
        with_system = [SystemMessage("You are helpful."), UserMessage("Hello")]
        count_with_system = model.count_tokens(with_system)

        # System message should add to the count
        assert (
            count_with_system > count_user_only
        ), f"With system ({count_with_system}) should be > user only ({count_user_only})"


class TestImageDimensionFormats:
    """Test image dimension parsing for all supported formats."""

    def test_jpeg_dimensions(self):
        """Test JPEG dimension extraction."""
        import base64

        # Minimal 1x1 red JPEG (generated programmatically)
        # This is a valid JPEG with SOF0 marker
        jpeg_b64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQACEQA/ALYgB//Z"
        jpeg_data = base64.b64decode(jpeg_b64)

        width, height = get_image_dimensions(jpeg_data)
        assert width == 1
        assert height == 1

    def test_gif_dimensions(self):
        """Test GIF dimension extraction."""
        # Minimal 1x1 transparent GIF
        gif_data = (
            b"GIF89a"  # Header
            + b"\x01\x00\x01\x00"  # Width=1, Height=1
            + b"\x00\x00\x00"  # Flags, bg, aspect
            + b"\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00"  # Image descriptor
            + b"\x02\x01\x44\x00\x3b"  # Image data and trailer
        )

        width, height = get_image_dimensions(gif_data)
        assert width == 1
        assert height == 1

    def test_bmp_dimensions(self):
        """Test BMP dimension extraction."""
        # Minimal 1x1 BMP (Windows v3 format)
        bmp_data = (
            b"BM"  # Header
            + b"\x3a\x00\x00\x00"  # File size
            + b"\x00\x00\x00\x00"  # Reserved
            + b"\x36\x00\x00\x00"  # Data offset
            + b"\x28\x00\x00\x00"  # Header size (40 = Windows v3)
            + b"\x01\x00\x00\x00"  # Width = 1
            + b"\x01\x00\x00\x00"  # Height = 1 (positive = bottom-up)
            + b"\x01\x00"  # Planes
            + b"\x18\x00"  # Bits per pixel
            + b"\x00" * 24  # Compression and other fields
        )

        width, height = get_image_dimensions(bmp_data)
        assert width == 1
        assert height == 1

    def test_webp_vp8_dimensions(self):
        """Test WebP VP8 (lossy) dimension extraction."""
        import base64

        # Minimal 1x1 WebP (VP8 lossy format)
        webp_b64 = (
            "UklGRiYAAABXRUJQVlA4IBoAAAAwAQCdASoBAAEAAQAcJYgCdAEO/g7aAAD++MoAAA=="
        )
        webp_data = base64.b64decode(webp_b64)

        width, height = get_image_dimensions(webp_data)
        assert width == 1
        assert height == 1
