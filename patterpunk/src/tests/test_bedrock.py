import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from patterpunk.llm.models.bedrock import BedrockModel
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.user import UserMessage
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
        "mistral.mistral-large-2402-v1:0",
    ],
)
def test_simple_bedrock(model_id):
    bedrock = BedrockModel(model_id=model_id, temperature=0.1, top_p=0.98)

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

    # Basic response checks
    assert response is not None, "Response should not be None"
    assert isinstance(
        response, str
    ), f"Response should be a string, got {type(response)}"
    assert len(response) > 0, "Response should not be empty"

    # Content validation - verify it answers the question about Canada
    assert (
        "canada" in response.lower()
    ), f"Response should mention Canada. Got: {response[:200]}"
    assert (
        "ottawa" in response.lower()
    ), f"Response should mention Ottawa as the capital. Got: {response[:200]}"

    # JSON format validation
    import json
    import re

    # Find JSON in the response (it might be embedded in other text)
    json_match = re.search(r'\{[^{}]*"country"[^{}]*\{[^{}]*\}[^{}]*\}', response)
    assert (
        json_match is not None
    ), f"Response should contain valid JSON format. Got: {response[:500]}"

    try:
        parsed_json = json.loads(json_match.group())
        assert "country" in parsed_json, "JSON should have 'country' key"
        assert "name" in parsed_json["country"], "JSON should have 'country.name' field"
        assert (
            "capital" in parsed_json["country"]
        ), "JSON should have 'country.capital' field"

        # Verify correct values
        country_name = parsed_json["country"]["name"].lower()
        assert (
            "canada" in country_name
        ), f"Country name should be Canada, got: {parsed_json['country']['name']}"

        capital_name = parsed_json["country"]["capital"].lower()
        assert (
            "ottawa" in capital_name
        ), f"Capital should be Ottawa, got: {parsed_json['country']['capital']}"
    except json.JSONDecodeError as e:
        pytest.fail(
            f"Failed to parse JSON from response: {e}. JSON string: {json_match.group()}"
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
    assert parsed_output.warranty_period is None
    assert parsed_output.competitor_comparison is None
    assert parsed_output.recommended is True


def test_simple_tool_calling():

    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city or location to get weather for
        """
        return f"The weather in {location} is sunny and 22°C"

    bedrock = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0.0, top_p=1.0
    )

    chat = Chat(model=bedrock).with_tools([get_weather])

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


def test_tool_calling():

    def calculate_area(length: float, width: float) -> str:
        area = length * width
        return f"The area is {area} square units"

    def get_math_fact(topic: str) -> str:
        facts = {
            "rectangle": "A rectangle has opposite sides that are equal and parallel",
            "area": "Area measures the amount of space inside a 2D shape",
            "geometry": "Geometry is one of the oldest mathematical sciences",
        }
        return facts.get(topic.lower(), "Mathematics is the language of the universe")

    bedrock = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0.0, top_p=1.0
    )

    chat = Chat(model=bedrock).with_tools([calculate_area, get_math_fact])

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

    # Check for calculate_area call
    area_calls = [tc for tc in tool_calls if tc["function"]["name"] == "calculate_area"]
    assert (
        len(area_calls) >= 1
    ), f"Expected calculate_area to be called, but got tools: {tool_names}"

    # Verify calculate_area arguments
    import json

    area_args = json.loads(area_calls[0]["function"]["arguments"])
    assert "length" in area_args, "calculate_area missing 'length' argument"
    assert "width" in area_args, "calculate_area missing 'width' argument"
    assert area_args["length"] == 5, f"Expected length=5, got {area_args['length']}"
    assert area_args["width"] == 3, f"Expected width=3, got {area_args['width']}"

    # Check for get_math_fact call (optional but expected)
    fact_calls = [tc for tc in tool_calls if tc["function"]["name"] == "get_math_fact"]
    if fact_calls:
        fact_args = json.loads(fact_calls[0]["function"]["arguments"])
        assert "topic" in fact_args, "get_math_fact missing 'topic' argument"
        assert (
            "rectangle" in fact_args["topic"].lower()
        ), f"Expected topic about rectangles, got {fact_args['topic']}"


@pytest.mark.parametrize(
    "model_id,region,thinking_config",
    [
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
        (
            "us.deepseek.r1-v1:0",
            "us-east-1",
            ThinkingConfig(effort="low"),
        ),
    ],
)
def test_thinking_mode_with_reasoning_models(model_id, region, thinking_config):

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

        assert response.latest_message is not None
        assert response.latest_message.content is not None

        content = response.latest_message.content

        if thinking_config.include_thoughts:
            assert "<thinking>" in content and "</thinking>" in content

        assert "391" in content or "three hundred ninety-one" in content.lower()

        assert any(
            keyword in content.lower()
            for keyword in ["step", "multiply", "calculate", "*", "×"]
        )

    except Exception as e:
        if ClientError and isinstance(e, ClientError):
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "AccessDeniedException":
                pytest.skip(
                    f"Access denied for inference profile model {model_id}. "
                    f"This model requires explicit access approval in AWS Bedrock console. "
                    f"Skipping test."
                )
        raise


@pytest.mark.parametrize(
    "model_id,thinking_config",
    [
        ("anthropic.claude-3-sonnet-20240229-v1:0", ThinkingConfig(token_budget=2000)),
        ("anthropic.claude-3-haiku-20240307-v1:0", ThinkingConfig(effort="low")),
    ],
)
def test_thinking_mode_unsupported_models_fail(model_id, thinking_config):
    """Verify that using ThinkingConfig with models that don't support thinking fails with a clear error."""

    bedrock = BedrockModel(
        model_id=model_id, temperature=0.1, top_p=0.98, thinking_config=thinking_config
    )

    chat = Chat(model=bedrock)

    # Should raise ValidationException when trying to use thinking mode with unsupported model
    with pytest.raises(ClientError) as exc_info:
        chat.add_message(
            UserMessage("What is 17 * 23? Please show your work.")
        ).complete()

    # Verify it's a validation error with a clear message
    error = exc_info.value
    assert (
        error.response["Error"]["Code"] == "ValidationException"
    ), f"Expected ValidationException but got {error.response['Error']['Code']}"

    # Check that the error message mentions the problematic field
    error_msg = str(error)

    # The error should mention which field caused the problem
    if thinking_config.token_budget is not None:
        assert (
            "reasoning_config" in error_msg
        ), f"Error should mention 'reasoning_config' for token_budget parameter. Got: {error_msg}"
    elif thinking_config.effort is not None:
        # With effort="low", Bedrock sends reasoning_effort which also causes validation error
        assert (
            "reasoning_effort" in error_msg or "reasoning_config" in error_msg
        ), f"Error should mention 'reasoning_effort' or 'reasoning_config' for effort parameter. Got: {error_msg}"

    # Verify the error message is clear about the issue
    assert (
        "not permitted" in error_msg or "Malformed" in error_msg
    ), f"Error message should clearly indicate the parameter is not permitted. Got: {error_msg}"


def test_thinking_mode_parameters():

    thinking_config_effort = ThinkingConfig(effort="high")
    bedrock_effort = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        thinking_config=thinking_config_effort,
    )

    thinking_params = bedrock_effort._get_thinking_params()
    assert "reasoning_effort" in thinking_params
    assert thinking_params["reasoning_effort"] == "high"

    thinking_config_budget = ThinkingConfig(token_budget=3000)
    bedrock_budget = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        thinking_config=thinking_config_budget,
    )

    thinking_params = bedrock_budget._get_thinking_params()
    assert "reasoning_config" in thinking_params
    assert thinking_params["reasoning_config"]["type"] == "enabled"
    assert thinking_params["reasoning_config"]["budget_tokens"] == 3000

    thinking_config_min = ThinkingConfig(token_budget=500)
    bedrock_min = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        thinking_config=thinking_config_min,
    )

    thinking_params = bedrock_min._get_thinking_params()
    assert thinking_params["reasoning_config"]["budget_tokens"] == 1024

    bedrock_none = BedrockModel(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    thinking_params = bedrock_none._get_thinking_params()
    assert thinking_params == {}


def test_multimodal_image():
    bedrock = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0.1, top_p=0.98
    )

    chat = Chat(model=bedrock)

    prepped_chat = chat.add_message(
        SystemMessage(
            """Carefully analyze the image. Answer in short, descriptive sentences. Answer questions clearly, directly and without flourish."""
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
    bedrock = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0.0, top_p=0.98
    )

    chat = Chat(model=bedrock)

    title = (
        chat.add_message(
            SystemMessage(
                """Create a single-line title for the given document. It needs to be descriptive and short, and not copied from the document"""
            )
        )
        .add_message(
            UserMessage(
                content=[
                    CacheChunk(
                        content="Please analyze this document and create a title.",
                        cacheable=False,
                    ),
                    MultimodalChunk.from_file(get_resource("research.pdf")),
                ]
            )
        )
        .complete()
        .latest_message.content
    )

    assert "bank of canada" in title.lower()
    assert "research" in title.lower()
    assert "2025" in title.lower()
