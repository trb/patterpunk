import json
import re
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from patterpunk.llm.models.bedrock import BedrockModel
from patterpunk.llm.output_limits import forget_output_limits, learned_output_limit
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.finish_reason import FinishReason
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.assistant import AssistantMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.chunks import CacheChunk, MultimodalChunk
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
                'What is the capital of Canada? Respond with a JSON object containing a "country" key, whose value is an object with two fields: "name" (the country name) and "capital" (the capital city name). Think out loud and work step by step. Show your work. Do this before you generate the JSON response.'
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

    # Find JSON in the response (it might be embedded in other text, and weaker
    # models may emit skeleton JSON during reasoning before the real answer)
    json_candidates = re.findall(r'\{[^{}]*"country"[^{}]*\{[^{}]*\}[^{}]*\}', response)
    assert (
        json_candidates
    ), f"Response should contain valid JSON format. Got: {response[:500]}"

    parsed_json = None
    for candidate in reversed(json_candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        country = parsed.get("country")
        if isinstance(country, dict) and country.get("name") and country.get("capital"):
            parsed_json = parsed
            break

    assert (
        parsed_json is not None
    ), f"Response should contain JSON with non-empty country.name and country.capital. Got: {response[:500]}"

    # Verify correct values
    country_name = parsed_json["country"]["name"].lower()
    assert (
        "canada" in country_name
    ), f"Country name should be Canada, got: {parsed_json['country']['name']}"

    capital_name = parsed_json["country"]["capital"].lower()
    assert (
        "ottawa" in capital_name
    ), f"Capital should be Ottawa, got: {parsed_json['country']['capital']}"


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
    """Test that tools are called and executed correctly with automatic execution."""

    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city or location to get weather for
        """
        return f"The weather in {location} is sunny and 22°C"

    # Use Claude Haiku 4.5 for reliable tool calling
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", temperature=0.0
    )

    chat = Chat(model=bedrock).with_tools([get_weather])

    system_msg = SystemMessage(
        "You are a helpful assistant that MUST ALWAYS use the provided tools. "
        "CRITICAL: You are REQUIRED to call the get_weather tool for ANY weather question. "
        "NEVER answer weather questions from your own knowledge. "
        "ALWAYS call the tool first, then respond based on the tool's output."
    )

    response = (
        chat.add_message(system_msg)
        .add_message(UserMessage("What's the weather in Paris?"))
        .complete()
    )

    # With automatic tool execution, the final message should be an AssistantMessage
    # containing the result from the tool
    assert response.latest_message is not None
    assert isinstance(response.latest_message, AssistantMessage), (
        f"Expected AssistantMessage but got {type(response.latest_message).__name__}. "
        f"Content: {response.latest_message.content}"
    )

    # The response should include the weather info from the tool
    response_lower = response.latest_message.content.lower()
    assert (
        "sunny" in response_lower or "22" in response_lower
    ), f"Expected weather info in response. Got: {response.latest_message.content}"

    # Verify a ToolCallMessage exists in the history (tool was called)
    tool_call_messages = [
        msg for msg in response.messages if isinstance(msg, ToolCallMessage)
    ]
    assert (
        len(tool_call_messages) >= 1
    ), "Expected at least one ToolCallMessage in history"


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

    # Use Claude Haiku 4.5 for reliable tool calling
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", temperature=0.0
    )

    chat = Chat(model=bedrock).with_tools([calculate_area, get_math_fact])

    system_msg = SystemMessage(
        "You are a geometry helper that MUST ALWAYS use the provided tools. "
        "CRITICAL: You are REQUIRED to call calculate_area for ANY area calculation. "
        "NEVER calculate area yourself - ALWAYS use the calculate_area tool. "
        "You must call the tool first, then explain the result."
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

    # With automatic tool execution, the final message should be an AssistantMessage
    # containing the result from the tool execution
    assert response.latest_message is not None
    assert isinstance(response.latest_message, AssistantMessage), (
        f"Expected AssistantMessage but got {type(response.latest_message).__name__}. "
        f"Content: {response.latest_message.content}"
    )

    # The response should include the area calculation result from the tool
    response_lower = response.latest_message.content.lower()
    assert (
        "15" in response_lower or "area" in response_lower
    ), f"Expected area calculation result in response. Got: {response.latest_message.content}"

    # Verify ToolCallMessage exists in the history (tool was called)
    tool_call_messages = [
        msg for msg in response.messages if isinstance(msg, ToolCallMessage)
    ]
    assert (
        len(tool_call_messages) >= 1
    ), "Expected at least one ToolCallMessage in history"

    # Get tool calls from history to verify correct parameters were used
    import json

    tool_calls = tool_call_messages[0].tool_calls
    tool_names = [tc.name for tc in tool_calls]

    # Check for calculate_area call
    area_calls = [tc for tc in tool_calls if tc.name == "calculate_area"]
    assert (
        len(area_calls) >= 1
    ), f"Expected calculate_area to be called, but got tools: {tool_names}"

    # Verify calculate_area arguments
    area_args = json.loads(area_calls[0].arguments)
    assert "length" in area_args, "calculate_area missing 'length' argument"
    assert "width" in area_args, "calculate_area missing 'width' argument"
    assert area_args["length"] == 5, f"Expected length=5, got {area_args['length']}"
    assert area_args["width"] == 3, f"Expected width=3, got {area_args['width']}"

    # Check for get_math_fact call (optional but expected)
    fact_calls = [tc for tc in tool_calls if tc.name == "get_math_fact"]
    if fact_calls:
        fact_args = json.loads(fact_calls[0].arguments)
        assert "topic" in fact_args, "get_math_fact missing 'topic' argument"
        assert (
            "rectangle" in fact_args["topic"].lower()
        ), f"Expected topic about rectangles, got {fact_args['topic']}"


@pytest.mark.parametrize(
    "model_id,region,thinking_config",
    [
        (
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us-east-1",
            ThinkingConfig(token_budget=2000),
        ),
        (
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
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
def test_thinking_mode_unsupported_models_adapts(model_id, thinking_config, caplog):
    """A ThinkingConfig on models without reasoning support is dropped with a
    WARNING and the request completes instead of raising ValidationException."""

    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id=model_id,
            temperature=0.1,
            top_p=0.98,
            thinking_config=thinking_config,
        )
    assert any(
        "does not accept reasoning parameters" in r.message for r in caplog.records
    )
    assert "additionalModelRequestFields" not in bedrock._build_converse_params([])

    chat = Chat(model=bedrock)
    response = chat.add_message(
        UserMessage("What is 17 * 23? Reply with just the number.")
    ).complete()
    assert "391" in response.latest_message.content


def test_thinking_mode_parameters():

    thinking_config_effort = ThinkingConfig(effort="high")
    bedrock_effort = BedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        thinking_config=thinking_config_effort,
    )

    thinking_params = bedrock_effort._get_thinking_params()
    assert thinking_params == {
        "reasoning_config": {"type": "enabled", "budget_tokens": 12000}
    }

    thinking_config_budget = ThinkingConfig(token_budget=3000)
    bedrock_budget = BedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        thinking_config=thinking_config_budget,
    )

    thinking_params = bedrock_budget._get_thinking_params()
    assert "reasoning_config" in thinking_params
    assert thinking_params["reasoning_config"]["type"] == "enabled"
    assert thinking_params["reasoning_config"]["budget_tokens"] == 3000

    thinking_config_min = ThinkingConfig(token_budget=500)
    bedrock_min = BedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        thinking_config=thinking_config_min,
    )

    thinking_params = bedrock_min._get_thinking_params()
    assert thinking_params["reasoning_config"]["budget_tokens"] == 1024

    bedrock_none = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
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


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ],
)
def test_cache_chunks(model_id):
    """Test that cache chunks work with Bedrock"""

    chat = Chat(model=BedrockModel(model_id=model_id, temperature=0.1, top_p=0.98))

    # Create a message with mixed cacheable and non-cacheable content
    large_context = """
    This is a large context document that should be cached for performance.
    It contains important information that will be referenced multiple times.
    """ * 100  # Make it larger to benefit from caching

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


def test_diagnostics_and_safety_filter_integration():
    """Live integration: confirms disable_safety_filters is accepted by Bedrock
    (no-op + debug log), finish_reason normalizes to STOP, and
    _provider.raw_finish_reason carries the Converse API 'end_turn' value."""
    response = (
        Chat(
            model=BedrockModel(
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                temperature=0.1,
            ),
            disable_safety_filters=True,
        )
        .add_message(UserMessage("Say hello in exactly one short sentence."))
        .complete()
        .latest_message
    )

    assert response.content
    assert len(response.content.strip()) > 0
    assert response.finish_reason == FinishReason.STOP
    assert response._provider.raw_finish_reason == "end_turn"


# =============================================================================
# Claude sampling-parameter filtering (shared claude_capabilities rules)
# =============================================================================


def test_claude_45_drops_top_p_with_warning(caplog):
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature=0.4,
        top_p=0.9,
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config == {"temperature": 0.4}
    assert any(
        "[BEDROCK]" in r.message and "Dropping top_p=0.9" in r.message
        for r in caplog.records
    )


def test_claude_45_without_top_p_is_silent(caplog):
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature=0.4,
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config == {"temperature": 0.4}
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_claude_5_omits_sampling_with_warning_for_custom_values(caplog):
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-opus-5-20260120-v1:0",
        temperature=0.4,
        top_p=0.9,
        max_tokens=2000,
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config == {"maxTokens": 2000}
    warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("temperature=0.4" in m and "top_p=0.9" in m for m in warning_messages)


def test_claude_5_at_defaults_is_silent_and_omits_inference_config(caplog):
    bedrock = BedrockModel(model_id="us.anthropic.claude-opus-5-20260120-v1:0")
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
        converse_params = bedrock._build_converse_params([])
    assert config == {}
    assert "inferenceConfig" not in converse_params
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_claude_3_sampling_passes_through(caplog):
    bedrock = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.1,
        top_p=0.98,
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config == {"temperature": 0.1, "topP": 0.98}
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_claude_45_thinking_budget_forces_temperature_and_max_tokens():
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature=1.0,
        max_tokens=2000,
        thinking_config=ThinkingConfig(token_budget=3000),
    )
    config = bedrock._build_inference_config()
    assert config["temperature"] == 1.0
    assert "topP" not in config
    assert config["maxTokens"] == 5000


def test_claude_45_thinking_effort_also_strips_sampling(caplog):
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        temperature=0.4,
        top_p=0.9,
        thinking_config=ThinkingConfig(effort="high"),
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config == {"temperature": 1.0, "maxTokens": 14000}
    warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "Coercing user-set temperature=0.4 to 1.0" in m for m in warning_messages
    )
    assert any("Dropping user-set top_p=0.9" in m for m in warning_messages)


def test_non_claude_models_keep_sampling_untouched(caplog):
    bedrock = BedrockModel(
        model_id="mistral.mistral-large-2402-v1:0",
        temperature=0.3,
        top_p=0.9,
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config == {"temperature": 0.3, "topP": 0.9}
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_unknown_claude_id_gets_newest_family_rules_with_warning(caplog):
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-futuristic-99",
        temperature=0.2,
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config == {}
    warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("Unrecognised Claude model id" in m for m in warning_messages)
    assert any("temperature=0.2" in m for m in warning_messages)


def test_thinking_config_ignored_on_non_reasoning_bedrock_models(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="meta.llama3-70b-instruct-v1:0",
            thinking_config=ThinkingConfig(effort="high"),
        )
    assert bedrock._get_thinking_params() == {}
    assert "additionalModelRequestFields" not in bedrock._build_converse_params([])
    assert any(
        "does not accept reasoning parameters" in r.message for r in caplog.records
    )


def test_thinking_config_ignored_below_claude_3_7(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            thinking_config=ThinkingConfig(token_budget=4000),
        )
    assert bedrock._get_thinking_params() == {}
    assert any(
        "does not accept reasoning parameters" in r.message for r in caplog.records
    )


def test_thinking_config_kept_on_claude_3_7(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            thinking_config=ThinkingConfig(token_budget=4000),
        )
    params = bedrock._get_thinking_params()
    assert params["reasoning_config"] == {"type": "enabled", "budget_tokens": 4000}
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_thinking_config_silent_on_deepseek(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.deepseek.r1-v1:0",
            thinking_config=ThinkingConfig(effort="high"),
        )
    assert bedrock._get_thinking_params() == {}
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_thinking_budget_zero_disables_reasoning():
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        thinking_config=ThinkingConfig(token_budget=0),
    )
    assert bedrock._get_thinking_params() == {}


def test_thinking_budget_below_minimum_warns_and_clamps(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            thinking_config=ThinkingConfig(token_budget=500),
        )
    params = bedrock._get_thinking_params()
    assert params["reasoning_config"]["budget_tokens"] == 1024
    assert any("Raising token_budget=500 to 1024" in r.message for r in caplog.records)


def warning_messages(caplog):
    return [r.message for r in caplog.records if r.levelname == "WARNING"]


def test_claude_5_effort_sends_adaptive_thinking_with_output_config(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-sonnet-5",
            thinking_config=ThinkingConfig(effort="low"),
        )
    assert bedrock._get_thinking_params() == {
        "reasoning_config": {"type": "adaptive"},
        "output_config": {"effort": "low"},
    }
    assert warning_messages(caplog) == []


def test_claude_5_effort_xhigh_passes_through():
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-sonnet-5",
        thinking_config=ThinkingConfig(effort="xhigh"),
    )
    assert bedrock._get_thinking_params()["output_config"] == {"effort": "xhigh"}


def test_claude_5_budget_coerced_to_adaptive_effort(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-sonnet-5",
            thinking_config=ThinkingConfig(token_budget=2000),
        )
    assert bedrock._get_thinking_params() == {
        "reasoning_config": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
    }
    assert any(
        "Coercing token_budget=2000 to effort='medium'" in m
        for m in warning_messages(caplog)
    )


def test_claude_5_budget_zero_sends_disabled_thinking_and_no_max_tokens(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-sonnet-5",
            thinking_config=ThinkingConfig(token_budget=0),
        )
    assert bedrock._get_thinking_params() == {"reasoning_config": {"type": "disabled"}}
    assert bedrock._build_inference_config() == {}
    assert warning_messages(caplog) == []


def test_claude_46_budget_zero_sends_disabled_thinking_and_keeps_temperature():
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-sonnet-4-6",
        temperature=0.25,
        thinking_config=ThinkingConfig(token_budget=0),
    )
    assert bedrock._get_thinking_params() == {"reasoning_config": {"type": "disabled"}}
    assert bedrock._build_inference_config() == {"temperature": 0.25}


def test_fable_budget_zero_sends_nothing_with_warning(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-fable-5-1",
            thinking_config=ThinkingConfig(token_budget=0),
        )
    assert bedrock._get_thinking_params() == {}
    assert any("cannot turn thinking off" in m for m in warning_messages(caplog))


def test_claude_5_include_thoughts_requests_summarized_display():
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-opus-5",
        thinking_config=ThinkingConfig(effort="high", include_thoughts=True),
    )
    assert bedrock._get_thinking_params() == {
        "reasoning_config": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
    }


def test_opus_46_v1_id_keeps_temperature_and_clamps_xhigh(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-opus-4-6-v1",
            temperature=0.25,
            thinking_config=ThinkingConfig(effort="xhigh"),
        )
        config = bedrock._build_inference_config()
    assert bedrock._get_thinking_params()["output_config"] == {"effort": "high"}
    assert config == {"temperature": 1.0}
    assert not any(
        "Unrecognised Claude model id" in m for m in warning_messages(caplog)
    )


def test_claude_5_adaptive_thinking_does_not_force_max_tokens():
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-sonnet-5",
        thinking_config=ThinkingConfig(effort="high"),
    )
    assert bedrock._build_inference_config() == {}


def test_claude_46_uses_adaptive_thinking_and_clamps_xhigh(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-6",
            thinking_config=ThinkingConfig(effort="xhigh"),
        )
    assert bedrock._get_thinking_params() == {
        "reasoning_config": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    assert any(
        "Clamping effort='xhigh' to 'high'" in m for m in warning_messages(caplog)
    )


def test_claude_46_accepts_max_effort(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-6",
            thinking_config=ThinkingConfig(effort="max"),
        )
    assert bedrock._get_thinking_params()["output_config"] == {"effort": "max"}
    assert warning_messages(caplog) == []


def test_haiku_45_effort_translated_to_budget(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            thinking_config=ThinkingConfig(effort="low"),
        )
    assert bedrock._get_thinking_params() == {
        "reasoning_config": {"type": "enabled", "budget_tokens": 1500}
    }
    assert warning_messages(caplog) == []


def test_haiku_45_xhigh_effort_clamped_to_high_budget(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            thinking_config=ThinkingConfig(effort="xhigh"),
        )
    assert bedrock._get_thinking_params()["reasoning_config"]["budget_tokens"] == 12000
    assert any(
        "Clamping effort='xhigh' to 'high'" in m for m in warning_messages(caplog)
    )


def test_gpt5_drops_sampling_with_warning(caplog):
    bedrock = BedrockModel(
        model_id="us.openai.gpt-5.6-luna",
        temperature=0.25,
        top_p=1.0,
        max_tokens=500,
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config == {"maxTokens": 500}
    assert any(
        "temperature=0.25" in m and "top_p=1.0" in m for m in warning_messages(caplog)
    )


@pytest.mark.parametrize(
    "model_id",
    ["in.openai.gpt-5.6-luna", "global.openai.gpt-5.6-luna", "openai.gpt-5.6-sol"],
)
def test_gpt5_detected_under_every_geo_prefix(model_id, caplog):
    bedrock = BedrockModel(model_id=model_id, temperature=0.25)
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config == {}
    assert any("temperature=0.25" in m for m in warning_messages(caplog))


def test_gpt5_at_defaults_is_silent_and_sends_no_reasoning_field(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(model_id="us.openai.gpt-5.6-luna")
        converse_params = bedrock._build_converse_params([])
    assert "inferenceConfig" not in converse_params
    assert "additionalModelRequestFields" not in converse_params
    assert warning_messages(caplog) == []


def test_gpt5_effort_sends_reasoning_object(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.openai.gpt-5.6-luna",
            thinking_config=ThinkingConfig(effort="low"),
        )
    assert bedrock._get_thinking_params() == {"reasoning": {"effort": "low"}}
    assert warning_messages(caplog) == []


def test_gpt5_budget_zero_sends_effort_none(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.openai.gpt-5.6-luna",
            thinking_config=ThinkingConfig(token_budget=0),
        )
    assert bedrock._get_thinking_params() == {"reasoning": {"effort": "none"}}
    assert warning_messages(caplog) == []


def test_gpt5_budget_coerced_to_effort(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.openai.gpt-5.6-luna",
            thinking_config=ThinkingConfig(token_budget=2000),
        )
    assert bedrock._get_thinking_params() == {"reasoning": {"effort": "medium"}}
    assert any(
        "Coercing token_budget=2000 to effort='medium'" in m
        for m in warning_messages(caplog)
    )


def test_gpt_oss_keeps_sampling_and_ignores_thinking_config(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="openai.gpt-oss-120b-1:0",
            temperature=0.3,
            top_p=0.9,
            thinking_config=ThinkingConfig(effort="low"),
        )
        config = bedrock._build_inference_config()
    assert config == {"temperature": 0.3, "topP": 0.9}
    assert bedrock._get_thinking_params() == {}
    assert any(
        "does not accept reasoning parameters" in m for m in warning_messages(caplog)
    )


def test_nova_2_lite_effort_sends_reasoning_config_and_keeps_sampling(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.amazon.nova-2-lite-v1:0",
            temperature=0.25,
            top_p=1.0,
            thinking_config=ThinkingConfig(effort="low"),
        )
        config = bedrock._build_inference_config()
    assert bedrock._get_thinking_params() == {
        "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}
    }
    assert config == {"temperature": 0.25, "topP": 1.0}
    assert warning_messages(caplog) == []


def test_nova_2_lite_budget_maps_to_effort(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.amazon.nova-2-lite-v1:0",
            thinking_config=ThinkingConfig(token_budget=2000),
        )
    assert bedrock._get_thinking_params()["reasoningConfig"] == {
        "type": "enabled",
        "maxReasoningEffort": "medium",
    }
    assert any("Coercing token_budget=2000" in m for m in warning_messages(caplog))


def test_nova_2_lite_budget_zero_sends_nothing():
    bedrock = BedrockModel(
        model_id="us.amazon.nova-2-lite-v1:0",
        thinking_config=ThinkingConfig(token_budget=0),
    )
    assert bedrock._get_thinking_params() == {}


def test_nova_2_lite_high_effort_drops_sampling_and_max_tokens(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.amazon.nova-2-lite-v1:0",
            temperature=0.25,
            top_p=1.0,
            max_tokens=500,
            thinking_config=ThinkingConfig(effort="max"),
        )
        config = bedrock._build_inference_config()
    assert bedrock._get_thinking_params()["reasoningConfig"]["maxReasoningEffort"] == (
        "high"
    )
    assert config == {}
    assert any(
        "temperature=0.25" in m and "top_p=1.0" in m and "max_tokens=500" in m
        for m in warning_messages(caplog)
    )


def test_nova_1_ignores_thinking_config_with_warning(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.amazon.nova-pro-v1:0",
            thinking_config=ThinkingConfig(effort="low"),
        )
    assert bedrock._get_thinking_params() == {}
    assert any(
        "does not accept reasoning parameters" in m for m in warning_messages(caplog)
    )


def test_claude_3_max_tokens_capped_to_output_limit(caplog):
    bedrock = BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens=8192,
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config["maxTokens"] == 4096
    assert any(
        "exceeds the 4096-token output limit" in r.message for r in caplog.records
    )


def test_claude_3_7_max_tokens_not_capped(caplog):
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        max_tokens=100000,
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config["maxTokens"] == 100000
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


@pytest.mark.parametrize(
    "model_id,requested,expected",
    [
        ("us.anthropic.claude-sonnet-5", 200000, 128000),
        ("us.anthropic.claude-opus-4-6-v1", 200000, 128000),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", 100000, 64000),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", 100000, 64000),
        ("us.openai.gpt-5.6-luna", 200000, 131072),
        ("us.amazon.nova-2-lite-v1:0", 100000, 65535),
    ],
)
def test_max_tokens_capped_to_model_output_limit(model_id, requested, expected, caplog):
    bedrock = BedrockModel(model_id=model_id, max_tokens=requested)
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config["maxTokens"] == expected
    assert any(
        f"exceeds the {expected}-token output limit" in m
        for m in warning_messages(caplog)
    )


def test_max_tokens_within_limit_passes_through_silently(caplog):
    bedrock = BedrockModel(model_id="us.anthropic.claude-sonnet-5", max_tokens=128000)
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = bedrock._build_inference_config()
    assert config["maxTokens"] == 128000
    assert warning_messages(caplog) == []


def test_max_tokens_unknown_family_is_not_capped():
    bedrock = BedrockModel(
        model_id="mistral.mistral-large-2402-v1:0", max_tokens=100000
    )
    assert bedrock._build_inference_config()["maxTokens"] == 100000


def _validation_error(message):
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": message}}, "Converse"
    )


def _limit_error(limit):
    return _validation_error(
        f"The maximum tokens you requested exceeds the model limit of {limit}. "
        f"Try again with a maximum tokens value that is lower than {limit}."
    )


def _converse_response(text="ok"):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
    }


@pytest.fixture
def fresh_output_limits():
    forget_output_limits()
    yield
    forget_output_limits()


def test_unknown_output_limit_is_learned_from_the_error_and_retried(
    fresh_output_limits, caplog
):
    bedrock = BedrockModel(model_id="meta.llama3-70b-instruct-v1:0", max_tokens=100000)
    bedrock.client = Mock()
    bedrock.client.converse.side_effect = [_limit_error(2048), _converse_response()]
    with caplog.at_level("WARNING", logger="patterpunk"):
        message = bedrock.generate_assistant_message([UserMessage("hi")])
    assert message.content == "ok"
    sent = [
        call.kwargs["inferenceConfig"]["maxTokens"]
        for call in bedrock.client.converse.call_args_list
    ]
    assert sent == [100000, 2048]
    assert any("caps output at 2048 tokens" in m for m in warning_messages(caplog))
    assert learned_output_limit("meta.llama3-70b-instruct-v1:0") == 2048

    later = BedrockModel(model_id="meta.llama3-70b-instruct-v1:0", max_tokens=100000)
    assert later._build_inference_config()["maxTokens"] == 2048


def test_learned_limit_lowers_a_thinking_budget_on_retry(fresh_output_limits):
    bedrock = BedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        thinking_config=ThinkingConfig(token_budget=100000),
    )
    bedrock.client = Mock()
    bedrock.client.converse.side_effect = [_limit_error(64000), _converse_response()]
    bedrock.generate_assistant_message([UserMessage("hi")])
    retried = bedrock.client.converse.call_args_list[1].kwargs
    assert retried["inferenceConfig"]["maxTokens"] == 64000
    assert retried["additionalModelRequestFields"]["reasoning_config"] == {
        "type": "enabled",
        "budget_tokens": 62000,
    }


def test_other_validation_errors_are_not_retried(fresh_output_limits):
    bedrock = BedrockModel(model_id="meta.llama3-70b-instruct-v1:0", max_tokens=100000)
    bedrock.client = Mock()
    bedrock.client.converse.side_effect = _validation_error(
        "The provided model identifier is invalid."
    )
    with pytest.raises(ClientError):
        bedrock.generate_assistant_message([UserMessage("hi")])
    assert bedrock.client.converse.call_count == 1
    assert learned_output_limit("meta.llama3-70b-instruct-v1:0") is None


def test_thinking_budget_lowered_to_fit_under_output_limit(caplog):
    with caplog.at_level("WARNING", logger="patterpunk"):
        bedrock = BedrockModel(
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            thinking_config=ThinkingConfig(token_budget=63000),
        )
        config = bedrock._build_inference_config()
    assert bedrock._get_thinking_params()["reasoning_config"]["budget_tokens"] == 62000
    assert config["maxTokens"] == 64000
    assert any(
        "Lowering the thinking budget to 62000" in m for m in warning_messages(caplog)
    )


def test_cache_point_is_a_separate_block_with_type_field():
    bedrock = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    content = bedrock._convert_content_to_bedrock_format(
        [CacheChunk("big document", cacheable=True)]
    )
    assert content == [
        {"text": "big document"},
        {"cachePoint": {"type": "default"}},
    ]


def test_cache_points_stripped_on_unsupported_models(caplog):
    bedrock = BedrockModel(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
    messages = [
        UserMessage([CacheChunk("big document", cacheable=True)]),
    ]
    with caplog.at_level("WARNING", logger="patterpunk"):
        converse_params = bedrock._build_converse_params(messages)
    content = converse_params["messages"][0]["content"]
    assert content == [{"text": "big document"}]
    assert any(
        "does not support prompt caching on Bedrock" in r.message
        for r in caplog.records
    )


def test_cache_points_kept_on_claude_3_5_haiku(caplog):
    bedrock = BedrockModel(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0")
    messages = [
        UserMessage([CacheChunk("big document", cacheable=True)]),
    ]
    with caplog.at_level("WARNING", logger="patterpunk"):
        converse_params = bedrock._build_converse_params(messages)
    content = converse_params["messages"][0]["content"]
    assert {"cachePoint": {"type": "default"}} in content
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_cache_points_trimmed_to_bedrock_limit_of_four(caplog):
    bedrock = BedrockModel(model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    messages = [
        UserMessage([CacheChunk(f"section {i}", cacheable=True) for i in range(6)]),
    ]
    with caplog.at_level("WARNING", logger="patterpunk"):
        converse_params = bedrock._build_converse_params(messages)
    content = converse_params["messages"][0]["content"]
    cache_points = [b for b in content if "cachePoint" in b]
    assert len(cache_points) == 4
    assert content[1] == {"text": "section 1"}
    assert "cachePoint" in content[5]
    assert any("at most 4 cache checkpoints" in r.message for r in caplog.records)
