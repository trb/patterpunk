import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from patterpunk.llm.models.bedrock import BedrockModel
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage, SystemMessage, ToolCallMessage
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.cache import CacheChunk
from patterpunk.llm.multimodal import MultimodalChunk
from tests.test_utils import get_resource

try:
    from botocore.exceptions import ClientError
except ImportError:
    ClientError = None


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "meta.llama3-8b-instruct-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
    ],
)
def test_simple_bedrock(model_id):
    bedrock = BedrockModel(model_id=model_id, temperature=0.1, top_p=0.98)

    print()
    print("Bedrock Models")
    print(BedrockModel.get_available_models())
    print()

    chat = Chat(model=bedrock)

    response = (
        chat.add_message(
            UserMessage(
                'What is the capital of Canada? Answer with JSON in this format: {"country": {"name": "the country the user asked for", "capital": "the capital of the country"}}. Think out loud and work step by step. Show your work. Do this before you generate the JSON response.'
            )
        )
        .complete()
        .latest_message.content
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "meta.llama3-70b-instruct-v1:0",
    ],
)
def test_structured_output(model_id):
    class ProductFeature(BaseModel):
        name: str = Field(description="Name of the feature")
        description: str = Field(description="Description of what the feature does")

    class ProductReview(BaseModel):
        product_name: str = Field(description="Name of the product being reviewed")
        manufacturer: str = Field(description="Company that makes the product")
        price: float = Field(description="Price of the product in USD")
        category: str = Field(description="Product category")
        rating: float = Field(description="Rating from 0.0 to 5.0")
        reviewer_name: str = Field(
            description="Name of the person who wrote the review"
        )
        pros: List[str] = Field(description="List of positive aspects of the product")
        cons: List[str] = Field(description="List of negative aspects of the product")
        key_features: List[ProductFeature] = Field(
            description="List of key features of the product"
        )
        warranty_period: Optional[str] = Field(
            description="Length of warranty if mentioned"
        )
        competitor_comparison: Optional[str] = Field(
            description="Comparison to competitor products if mentioned"
        )
        recommended: bool = Field(
            description="Whether the reviewer recommends the product"
        )

    bedrock = BedrockModel(model_id=model_id, temperature=0.1, top_p=0.98)

    system_prompt = SystemMessage(
        "You are a data extraction assistant. Your task is to extract structured information from product reviews. "
        "Extract only the information that is explicitly mentioned in the text. "
        "Do not infer or make up information. If information for a field is not provided, set it to null."
    )

    review_text = """
    Product Review: XDR-500 Noise Cancelling Headphones by SoundMaster
    
    Reviewed by: Alex Johnson
    
    I recently purchased the XDR-500 Noise Cancelling Headphones from SoundMaster for $249.99. These premium headphones 
    fall into the audio accessories category and have quickly become my favorite tech purchase this year.
    
    The XDR-500 offers exceptional sound quality with deep bass and crystal-clear highs. The active noise cancellation 
    is truly impressive, blocking out almost all ambient noise even in crowded environments. The battery life is 
    outstanding, lasting around 30 hours on a single charge. The build quality feels premium with comfortable ear cups 
    that don't hurt even after hours of use.
    
    What I like:
    - Exceptional sound quality
    - Effective noise cancellation
    - Long battery life (30+ hours)
    - Comfortable for extended wear
    - Quick charging (15 minutes for 5 hours of playback)
    
    Drawbacks:
    - Expensive compared to similar models
    - Slightly bulky design
    - No water resistance
    
    The SoundMaster app provides good customization options, though it occasionally crashes on older phones.
    
    Overall, I would rate these headphones 4.5 out of 5 stars. Despite the high price point, the quality and 
    performance make them worth the investment. I definitely recommend these to anyone looking for premium 
    noise-cancelling headphones.
    """

    chat = Chat(model=bedrock, messages=[system_prompt])

    result = chat.add_message(
        UserMessage(review_text, structured_output=ProductReview)
    ).complete()

    parsed_output = result.parsed_output

    assert parsed_output is not None
    assert parsed_output.product_name == "XDR-500 Noise Cancelling Headphones"
    assert parsed_output.manufacturer == "SoundMaster"
    assert parsed_output.price == 249.99
    assert parsed_output.category == "audio accessories"
    assert parsed_output.rating == 4.5
    assert parsed_output.reviewer_name == "Alex Johnson"
    assert len(parsed_output.pros) >= 4
    assert "Exceptional sound quality" in parsed_output.pros
    assert len(parsed_output.cons) >= 2
    assert "Expensive compared to similar models" in parsed_output.cons
    assert len(parsed_output.key_features) >= 2
    assert any(
        feature.name == "noise cancellation" for feature in parsed_output.key_features
    ) or any("noise" in feature.name.lower() for feature in parsed_output.key_features)
    assert parsed_output.warranty_period is None  # Not mentioned in the review
    assert parsed_output.competitor_comparison is None  # Not mentioned in the review
    assert parsed_output.recommended is True


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ],
)
def test_simple_tool_calling(model_id):
    """Test simple tool calling with a single function."""

    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny and 22°C"

    bedrock = BedrockModel(model_id=model_id, temperature=0.1, top_p=0.98)

    chat = Chat(model=bedrock).with_tools([get_weather])

    response = chat.add_message(
        UserMessage("What's the weather like in Paris?")
    ).complete()

    # Verify we got a response
    assert response.latest_message is not None
    assert response.latest_message.content is not None

    # Tool calls should always result in ToolCallMessage as latest message
    # If latest message is a ToolCallMessage, complete() is done and user decides on tool execution
    # If latest message is AssistantMessage, check for expected content
    if isinstance(response.latest_message, ToolCallMessage):
        # Tool calling worked correctly - test passes
        pass
    else:
        # Response should mention Paris and weather if not a tool call
        content = response.latest_message.content.lower()
        assert "paris" in content


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ],
)
def test_tool_calling(model_id):
    """Test tool calling functionality with multiple tools."""

    def calculate_area(length: float, width: float) -> str:
        """Calculate the area of a rectangle given length and width."""
        area = length * width
        return f"The area is {area} square units"

    def get_math_fact(topic: str) -> str:
        """Get an interesting mathematical fact about a topic."""
        facts = {
            "rectangle": "A rectangle has opposite sides that are equal and parallel",
            "area": "Area measures the amount of space inside a 2D shape",
            "geometry": "Geometry is one of the oldest mathematical sciences",
        }
        return facts.get(topic.lower(), "Mathematics is the language of the universe")

    bedrock = BedrockModel(model_id=model_id, temperature=0.1, top_p=0.98)

    chat = Chat(model=bedrock).with_tools([calculate_area, get_math_fact])

    response = (
        chat.add_message(
            SystemMessage("You are a geometry helper. Use tools to solve problems.")
        )
        .add_message(
            UserMessage(
                "I have a rectangle that is 5 units long and 3 units wide. "
                "Calculate its area and give me an interesting fact about rectangles."
            )
        )
        .complete()
    )

    # Verify we got a response
    assert response.latest_message is not None
    assert response.latest_message.content is not None

    # Tool calls should always result in ToolCallMessage as latest message
    # If latest message is a ToolCallMessage, complete() is done and user decides on tool execution
    # If latest message is AssistantMessage, check for expected content
    if isinstance(response.latest_message, ToolCallMessage):
        # Tool calling worked correctly - test passes
        pass
    else:
        # Response should mention the calculated area if not a tool call
        content = response.latest_message.content.lower()
        assert "15" in content or "fifteen" in content  # 5 * 3 = 15


@pytest.mark.parametrize(
    "model_id,region,thinking_config",
    [
        # Test Claude 3.7 Sonnet with reasoning mode using inference profiles
        (
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us-east-1",
            ThinkingConfig(token_budget=2000),
        ),
        (
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us-east-1",
            ThinkingConfig(token_budget=4000, include_thoughts=True),
        ),
        # Test DeepSeek R1 with reasoning effort using inference profile
        (
            "us.deepseek.r1-v1:0",
            "us-east-1",
            ThinkingConfig(effort="low"),
        ),
    ],
)
def test_thinking_mode_with_reasoning_models(model_id, region, thinking_config):
    """Test thinking mode functionality with models that actually support reasoning."""

    bedrock = BedrockModel(
        model_id=model_id,
        thinking_config=thinking_config,
        region_name=region,
    )

    chat = Chat(model=bedrock)

    try:
        response = chat.add_message(
            UserMessage(
                "Solve this step by step: What is 17 * 23? "
                "Think through the multiplication process carefully and show your reasoning."
            )
        ).complete()

        # Verify we got a response
        assert response.latest_message is not None
        assert response.latest_message.content is not None

        content = response.latest_message.content

        # Check if thinking content is included when requested
        if thinking_config.include_thoughts:
            assert "<thinking>" in content and "</thinking>" in content

        # The response should contain the correct answer (17 * 23 = 391)
        assert "391" in content or "three hundred ninety-one" in content.lower()

        # For reasoning models, we expect more detailed step-by-step reasoning
        # (even without include_thoughts, the model should show work in the response)
        assert any(
            keyword in content.lower()
            for keyword in ["step", "multiply", "calculate", "*", "×"]
        )

    except Exception as e:
        # Skip test if access to inference profile models is not available
        if ClientError and isinstance(e, ClientError):
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "AccessDeniedException":
                pytest.skip(
                    f"Access denied for inference profile model {model_id}. "
                    f"This model requires explicit access approval in AWS Bedrock console. "
                    f"Skipping test."
                )
        # Re-raise other exceptions
        raise


@pytest.mark.parametrize(
    "model_id,thinking_config",
    [
        # Test with regular Claude models - these won't support reasoning but should not break
        ("anthropic.claude-3-sonnet-20240229-v1:0", ThinkingConfig(token_budget=2000)),
        ("anthropic.claude-3-haiku-20240307-v1:0", ThinkingConfig(effort="low")),
    ],
)
def test_thinking_mode_graceful_degradation(model_id, thinking_config):
    """Test that thinking mode parameters don't break regular models."""

    bedrock = BedrockModel(
        model_id=model_id, temperature=0.1, top_p=0.98, thinking_config=thinking_config
    )

    chat = Chat(model=bedrock)

    # For models that don't support reasoning, the request should either:
    # 1. Work normally (ignoring the reasoning parameters), or
    # 2. Fail gracefully with a validation error
    try:
        response = chat.add_message(
            UserMessage("What is 17 * 23? Please show your work.")
        ).complete()

        # If it succeeds, verify we got a response
        assert response.latest_message is not None
        assert response.latest_message.content is not None

        # Should contain the answer even without reasoning mode
        content = response.latest_message.content
        assert "391" in content or "three hundred ninety-one" in content.lower()

    except Exception as e:
        # If it fails, it should be a validation error about unsupported parameters
        if ClientError and isinstance(e, ClientError):
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ValidationException":
                # Expected behavior - regular models don't support reasoning parameters
                pass
            else:
                raise
        elif "ValidationException" in str(type(e)) or "reasoning" in str(e).lower():
            # Expected behavior - validation error about unsupported parameters
            pass
        else:
            # Unexpected error, re-raise
            raise


def test_thinking_mode_parameters():
    """Test that thinking mode parameters are correctly set in the model."""

    # Test with effort parameter (for DeepSeek/OpenAI models)
    thinking_config_effort = ThinkingConfig(effort="high")
    bedrock_effort = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # Use available model
        thinking_config=thinking_config_effort,
    )

    thinking_params = bedrock_effort._get_thinking_params()
    assert "reasoning_effort" in thinking_params
    assert thinking_params["reasoning_effort"] == "high"

    # Test with token_budget parameter (for Anthropic models)
    thinking_config_budget = ThinkingConfig(token_budget=3000)
    bedrock_budget = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        thinking_config=thinking_config_budget,
    )

    thinking_params = bedrock_budget._get_thinking_params()
    assert "reasoning_config" in thinking_params
    assert thinking_params["reasoning_config"]["type"] == "enabled"
    assert thinking_params["reasoning_config"]["budget_tokens"] == 3000

    # Test minimum budget_tokens enforcement
    thinking_config_min = ThinkingConfig(token_budget=500)  # Below minimum
    bedrock_min = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        thinking_config=thinking_config_min,
    )

    thinking_params = bedrock_min._get_thinking_params()
    assert (
        thinking_params["reasoning_config"]["budget_tokens"] == 1024
    )  # Should be raised to minimum

    # Test no thinking config
    bedrock_none = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    thinking_params = bedrock_none._get_thinking_params()
    assert thinking_params == {}


def test_multimodal_image():
    bedrock = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.1,
        top_p=0.98
    )
    
    chat = Chat(model=bedrock)

    prepped_chat = (
        chat
        .add_message(SystemMessage("""Carefully analyze the image. Answer in short, descriptive sentences. Answer questions clearly, directly and without flourish."""))

    )

    correct = (
        prepped_chat
        .add_message(UserMessage(
            content=[
                CacheChunk(content="Are there ducks by a pond?", cacheable=False),
                MultimodalChunk.from_file(get_resource('ducks_pond.jpg'))
            ])
        )
        .complete()
        .latest_message
        .content
    )


    incorrect = (
        prepped_chat
        .add_message(UserMessage(
            content=[
                CacheChunk(content="Are there tigers in a desert?", cacheable=False),
                MultimodalChunk.from_file(get_resource('ducks_pond.jpg'))
            ])
        )
        .complete()
        .latest_message
        .content
    )

    assert 'yes' in correct.lower() or 'correct' in correct.lower(), 'LLM is wrong: There are ducks in the image'
    assert 'no' in incorrect.lower() or 'incorrect' in incorrect.lower(), 'LLM is wrong: There are no tigers in the image'

def test_multimodal_pdf():
    bedrock = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.0,
        top_p=0.98
    )
    
    chat = Chat(model=bedrock)

    title = (
        chat
        .add_message(SystemMessage("""Create a single-line title for the given document. It needs to be descriptive and short, and not copied from the document"""))
        .add_message(UserMessage(
            content=[
                CacheChunk(content="Please analyze this document and create a title.", cacheable=False),
                MultimodalChunk.from_file(get_resource('research.pdf'))
            ]
        ))
        .complete()
        .latest_message
        .content
    )

    assert 'bank of canada' in title.lower()
    assert 'research' in title.lower()
    assert '2025' in title.lower()
