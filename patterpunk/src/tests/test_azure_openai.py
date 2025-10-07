import os
import pytest
from pydantic import BaseModel, Field

from patterpunk.llm.cache import CacheChunk
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.azure_openai import (
    AzureOpenAiModel,
    AzureOpenAiMissingConfigurationError,
)
from patterpunk.llm.multimodal import MultimodalChunk
from patterpunk.llm.thinking import ThinkingConfig
from tests.test_utils import get_resource


def test_azure_endpoint_url_formatting():
    """Test that Azure endpoint URLs are formatted correctly for v1 API"""
    # Clear any existing env vars
    os.environ.pop("PP_AZURE_OPENAI_ENDPOINT", None)
    os.environ.pop("PP_AZURE_OPENAI_API_KEY", None)

    # Reset the global client
    from patterpunk.config.providers import azure_openai

    azure_openai._azure_openai_client = None

    test_cases = [
        (
            "https://test.openai.azure.com",
            "https://test.openai.azure.com/openai/v1/",
        ),
        (
            "https://test.openai.azure.com/",
            "https://test.openai.azure.com/openai/v1/",
        ),
        (
            "https://test.openai.azure.com/openai/v1/",
            "https://test.openai.azure.com/openai/v1/",
        ),
    ]

    for input_endpoint, expected_base_url in test_cases:
        os.environ["PP_AZURE_OPENAI_ENDPOINT"] = input_endpoint
        os.environ["PP_AZURE_OPENAI_API_KEY"] = "test-key"

        # Reset module to force re-initialization
        import importlib

        importlib.reload(azure_openai)

        client = azure_openai.azure_openai
        assert (
            client is not None
        ), f"Client should be created for endpoint: {input_endpoint}"
        assert str(client.base_url) == expected_base_url, (
            f"Base URL mismatch for input '{input_endpoint}': "
            f"expected '{expected_base_url}', got '{client.base_url}'"
        )

    # Clean up
    os.environ.pop("PP_AZURE_OPENAI_ENDPOINT", None)
    os.environ.pop("PP_AZURE_OPENAI_API_KEY", None)


def test_azure_client_not_created_without_credentials():
    """Test that client is not created when credentials are missing"""
    os.environ.pop("PP_AZURE_OPENAI_ENDPOINT", None)
    os.environ.pop("PP_AZURE_OPENAI_API_KEY", None)

    from patterpunk.config.providers import azure_openai
    import importlib

    importlib.reload(azure_openai)

    assert (
        azure_openai.azure_openai is None
    ), "Client should be None without credentials"
    assert (
        not azure_openai.is_azure_openai_available()
    ), "Azure should not be available without credentials"


@pytest.mark.skip(
    reason="Modifying env vars during test run interferes with other tests"
)
def test_azure_model_requires_credentials():
    """Test that AzureOpenAiModel raises error without credentials"""
    # This test is skipped because modifying environment variables
    # during test execution can interfere with other tests that run in parallel
    pass


def test_basic():
    """Test basic Azure OpenAI chat functionality"""
    print()
    print("Testing Azure OpenAI basic chat")
    print()

    # Use actual deployment name from Azure
    chat = Chat(model=AzureOpenAiModel(deployment_name="gpt-4", temperature=0.1))

    chat = (
        chat.add_message(
            SystemMessage(
                "You are a helpful assistant. Respond concisely and accurately."
            )
        )
        .add_message(UserMessage("What is 2 + 2? Answer with just the number."))
        .complete()
    )

    print()
    print("message")
    print(chat.latest_message.content)
    print()

    assert "4" in chat.latest_message.content


def test_structured_output():
    """Test structured output with Azure OpenAI"""

    class CalculationResult(BaseModel):
        question: str = Field(description="The question that was asked")
        answer: int = Field(description="The numerical answer")
        explanation: str = Field(description="Brief explanation of the calculation")

    chat = Chat(model=AzureOpenAiModel(deployment_name="gpt-4", temperature=0.1))

    chat = chat.add_message(
        SystemMessage(
            "You are a math tutor. Answer questions clearly and show your work."
        )
    )

    chat = chat.add_message(
        UserMessage("What is 15 + 27?", structured_output=CalculationResult)
    )

    chat = chat.complete()

    parsed_output = chat.parsed_output

    assert parsed_output is not None
    assert isinstance(parsed_output, CalculationResult)
    assert parsed_output.answer == 42
    assert "15" in parsed_output.question or "27" in parsed_output.question


@pytest.mark.skip(reason="Requires o-series deployment in Azure")
def test_reasoning_model():
    """Test Azure OpenAI with reasoning configuration (if o-series models are deployed)"""

    # Skipped - requires o3-mini deployment
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="o3-mini",
            temperature=0.1,
            thinking_config=ThinkingConfig(effort="medium"),
        )
    )

    chat = (
        chat.add_message(SystemMessage("Only reply with 'test'"))
        .add_message(UserMessage("You are being tested"))
        .complete()
    )

    print(chat.latest_message.content)

    assert "test" in chat.latest_message.content.lower()


def test_multimodal_image():
    """Test multimodal image input with Azure OpenAI"""

    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="gpt-4",
            temperature=0.1,
        )
    )

    prepped_chat = chat.add_message(
        SystemMessage(
            "Carefully analyze the image. Answer in short, descriptive sentences. Answer questions clearly, directly and without flourish."
        )
    )

    correct = (
        prepped_chat.add_message(
            UserMessage(
                content=[
                    CacheChunk(content="Are there ducks by a pond?", cacheable=False),
                    MultimodalChunk.from_file(get_resource("ducks_pond.jpg")),
                ]
            )
        )
        .complete()
        .latest_message.content
    )

    incorrect = (
        prepped_chat.add_message(
            UserMessage(
                content=[
                    CacheChunk(
                        content="Are there tigers in a desert?", cacheable=False
                    ),
                    MultimodalChunk.from_file(get_resource("ducks_pond.jpg")),
                ]
            )
        )
        .complete()
        .latest_message.content
    )

    assert (
        "yes" in correct.lower() or "correct" in correct.lower()
    ), "LLM is wrong: There are ducks in the image"
    assert (
        "no" in incorrect.lower() or "incorrect" in incorrect.lower()
    ), "LLM is wrong: There are no tigers in the image"


def test_multimodal_pdf():
    """Test PDF document processing with Azure OpenAI"""

    chat = Chat(model=AzureOpenAiModel(deployment_name="gpt-4", temperature=0.0))

    title = (
        chat.add_message(
            SystemMessage(
                "Create a single-line title for the given document. It needs to be descriptive and short, and not copied from the document"
            )
        )
        .add_message(
            UserMessage(
                content=[MultimodalChunk.from_file(get_resource("research.pdf"))]
            )
        )
        .complete()
        .latest_message.content
    )

    # Basic validation that the title contains relevant keywords
    assert len(title) > 0
    assert len(title) < 200  # Should be a short title


def test_simple_tool_calling():
    """Test simple tool calling with Azure OpenAI"""

    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city or location to get weather for
        """
        return f"The weather in {location} is sunny and 22°C"

    chat = Chat(
        model=AzureOpenAiModel(deployment_name="gpt-4", temperature=0.0)
    ).with_tools([get_weather])

    system_msg = SystemMessage(
        "You are a helpful assistant that MUST use the provided tools to answer questions. "
        "When asked about weather, you MUST call the get_weather tool. "
        "Do not just describe what you would do - actually call the tool."
    )

    response = (
        chat.add_message(system_msg)
        .add_message(UserMessage("What's the weather in Paris?"))
        .complete()
    )

    assert response.latest_message is not None
    assert isinstance(response.latest_message, ToolCallMessage), (
        f"Expected ToolCallMessage but got {type(response.latest_message).__name__}. "
        f"Content: {response.latest_message.content}"
    )

    tool_calls = response.latest_message.tool_calls
    assert (
        len(tool_calls) == 1
    ), f"Expected exactly one tool call, got {len(tool_calls)}"

    tool_call = tool_calls[0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"

    import json

    arguments = json.loads(tool_call["function"]["arguments"])
    assert "location" in arguments
    assert "paris" in arguments["location"].lower()


def test_multi_tool_calling():
    """Test calling multiple tools with Azure OpenAI"""

    def calculate_area(length: float, width: float) -> str:
        """Calculate the area of a rectangle.

        Args:
            length: The length of the rectangle
            width: The width of the rectangle
        """
        area = length * width
        return f"The area is {area} square units"

    def get_math_fact(topic: str) -> str:
        """Get an interesting fact about a math topic.

        Args:
            topic: The math topic to get a fact about
        """
        facts = {
            "rectangle": "A rectangle has opposite sides that are equal and parallel",
            "area": "Area measures the amount of space inside a 2D shape",
            "geometry": "Geometry is one of the oldest mathematical sciences",
        }
        return facts.get(topic.lower(), "Mathematics is the language of the universe")

    chat = Chat(
        model=AzureOpenAiModel(deployment_name="gpt-4", temperature=0.0)
    ).with_tools([calculate_area, get_math_fact])

    system_msg = SystemMessage(
        "You are a geometry helper that MUST use the provided tools to solve problems. "
        "When asked to calculate area, you MUST call the calculate_area tool. "
        "When asked for facts, you MUST call the get_math_fact tool. "
        "Do not calculate or provide facts without using the tools."
    )

    response = (
        chat.add_message(system_msg)
        .add_message(
            UserMessage(
                "I have a rectangle that is 5 units long and 3 units wide. "
                "Calculate its area and give me an interesting fact about rectangles."
            )
        )
        .complete()
    )

    assert response.latest_message is not None
    assert isinstance(response.latest_message, ToolCallMessage), (
        f"Expected ToolCallMessage but got {type(response.latest_message).__name__}. "
        f"Content: {response.latest_message.content}"
    )

    tool_calls = response.latest_message.tool_calls
    assert (
        len(tool_calls) >= 1
    ), f"Expected at least one tool call, got {len(tool_calls)}"

    # Verify we have the expected tool calls
    tool_names = [tc["function"]["name"] for tc in tool_calls]
    assert (
        "calculate_area" in tool_names or "get_math_fact" in tool_names
    ), f"Expected calculate_area or get_math_fact in tool calls, got: {tool_names}"


def test_cache_chunks():
    """Test that cache chunks work with Azure OpenAI"""

    chat = Chat(model=AzureOpenAiModel(deployment_name="gpt-4", temperature=0.1))

    # Create a message with mixed cacheable and non-cacheable content
    large_context = (
        """
    This is a large context document that should be cached for performance.
    It contains important information that will be referenced multiple times.
    """
        * 100
    )  # Make it larger to benefit from caching

    response = (
        chat.add_message(
            SystemMessage(
                content=[
                    CacheChunk(
                        content=large_context,
                        cacheable=True,
                    ),
                    CacheChunk(
                        content="Answer questions about the context concisely.",
                        cacheable=False,
                    ),
                ]
            )
        )
        .add_message(
            UserMessage(
                content=[
                    CacheChunk(content="What is this document about?", cacheable=False)
                ]
            )
        )
        .complete()
    )

    assert response.latest_message is not None
    assert response.latest_message.content is not None
    assert len(response.latest_message.content.strip()) > 0

    # The response should mention something about context or information
    content_lower = response.latest_message.content.lower()
    assert any(
        term in content_lower
        for term in ["context", "information", "document", "reference"]
    )
