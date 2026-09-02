from typing import Optional, List

from pydantic import BaseModel, Field

from patterpunk.llm.chat.core import Chat
from patterpunk.llm.finish_reason import FinishReason
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.chunks import CacheChunk, MultimodalChunk, TextChunk
from tests.test_utils import get_resource


def test_basic():
    model = GoogleModel(
        model="gemini-2.5-flash", location="us-central1", temperature=1.1
    )

    response = (
        Chat(model=model)
        .add_message(SystemMessage("Speak like a pirate"))
        .add_message(
            UserMessage("Write three random, unrelated sentences about any old subject")
        )
        .complete()
        .latest_message
    )

    assert len(response.content) > 50


def test_structured_output():
    class BookInfo(BaseModel):
        title: str
        author: str
        publication_year: int
        isbn: Optional[str] = None
        genres: Optional[List[str]] = None
        page_count: Optional[int] = None
        is_bestseller: Optional[bool] = None

    class ThoughtfulBookResponse(BaseModel):
        requirements: str = Field(description="List the requirements for this request")
        thoughts: str = Field(
            description="Think out loud about how you will complete the request. Be careful to catch edge cases and subtleties. Length: thorough and lengthy, detail-oriented"
        )
        book_info: BookInfo = Field(
            description="The BookInfo structure representing the requested data. Follow the provided schema carefully. Do not infer fields, only include information that is present in the source message"
        )

    model = GoogleModel(
        model="gemini-2.5-flash", location="us-central1", temperature=0.1
    )
    chat = Chat(model=model)

    sample_text = """
    "To Kill a Mockingbird" is a novel by Harper Lee published in 1960. 
    It was immediately successful, winning the Pulitzer Prize, and has 
    become a classic of modern American literature. The plot and characters 
    are loosely based on Lee's observations of her family, her neighbors 
    and an event that occurred near her hometown of Monroeville, Alabama, in 1936, 
    when she was 10 years old.
    """

    chat = chat.add_message(SystemMessage("""
            Extract information about the book from the provided text.
            Include the title, author, and publication year.
            If available, also extract the ISBN, genres, page count, and whether it's a bestseller.
            
            Strictly extract information about the book from the provided text and do not infer information.
            """))

    chat = chat.add_message(
        UserMessage(sample_text, structured_output=ThoughtfulBookResponse)
    )

    chat = chat.complete()

    parsed_output: ThoughtfulBookResponse = chat.parsed_output

    assert parsed_output.book_info.author == "Harper Lee"
    assert parsed_output.book_info.title == "To Kill a Mockingbird"
    # Note: Gemini 2.5 may infer genres from context, so we don't assert None
    assert parsed_output.book_info.publication_year == 1960


def test_available_models():
    us_models = GoogleModel.get_available_models(location="us-central1")
    assert "gemini-2.5-flash" in us_models
    assert "gemini-2.5-pro" in us_models
    assert "gemini-2.0-flash" in us_models

    # Test a different US region
    us_east_models = GoogleModel.get_available_models(location="us-east4")
    assert len(us_east_models) > 0


def test_tool_calling():
    from patterpunk.llm.chat.core import Chat

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

    model = GoogleModel(
        model="gemini-2.5-flash", location="us-central1", temperature=0.1
    )

    chat = Chat(model=model).with_tools([calculate_area, get_math_fact])

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

    assert response.latest_message is not None
    assert response.latest_message.content is not None

    if isinstance(response.latest_message, ToolCallMessage):
        pass
    else:
        content = response.latest_message.content.lower()
        assert "15" in content or "fifteen" in content


def test_simple_tool_calling():
    from patterpunk.llm.chat.core import Chat

    def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny and 22°C"

    model = GoogleModel(
        model="gemini-2.5-flash", location="us-central1", temperature=0.1
    )

    chat = Chat(model=model).with_tools([get_weather])

    response = chat.add_message(
        UserMessage("What's the weather like in Paris?")
    ).complete()

    assert response.latest_message is not None
    assert response.latest_message.content is not None

    if isinstance(response.latest_message, ToolCallMessage):
        pass
    else:
        content = response.latest_message.content.lower()
        assert "paris" in content


def test_thinking_mode_fixed_budget():
    from patterpunk.llm.thinking import ThinkingConfig

    model = GoogleModel(
        model="gemini-2.5-flash",
        location="us-central1",
        temperature=0.1,
        thinking_config=ThinkingConfig(token_budget=1024),
    )

    response = (
        Chat(model=model)
        .add_message(SystemMessage("You are a helpful assistant."))
        .add_message(UserMessage("Solve this step by step: What is 23 * 47?"))
        .complete()
        .latest_message
    )

    assert response.content is not None
    assert len(response.content) > 0
    content = response.content.lower()
    assert "1081" in content or "23" in content or "47" in content
    assert (
        response.thinking_token_count is not None and response.thinking_token_count > 0
    )


def test_thinking_mode_dynamic_budget():
    from patterpunk.llm.thinking import ThinkingConfig

    model = GoogleModel(
        model="gemini-2.5-flash",
        location="us-central1",
        temperature=0.1,
        thinking_config=ThinkingConfig(effort="high"),
    )

    response = (
        Chat(model=model)
        .add_message(SystemMessage("You are a helpful assistant."))
        .add_message(
            UserMessage(
                "Explain the concept of recursion in programming with an example."
            )
        )
        .complete()
        .latest_message
    )

    assert response.content is not None
    assert len(response.content) > 100
    content = response.content.lower()
    assert "recursion" in content
    assert (
        response.thinking_token_count is not None and response.thinking_token_count > 0
    )


def test_thinking_mode_disabled():
    from patterpunk.llm.thinking import ThinkingConfig

    model = GoogleModel(
        model="gemini-2.5-flash",
        location="us-central1",
        temperature=0.1,
        thinking_config=ThinkingConfig(token_budget=0),
    )

    response = (
        Chat(model=model)
        .add_message(SystemMessage("You are a helpful assistant."))
        .add_message(UserMessage("What is the capital of France?"))
        .complete()
        .latest_message
    )

    assert response.content is not None
    assert len(response.content) > 0
    content = response.content.lower()
    assert "paris" in content
    assert response.thinking_token_count is None or response.thinking_token_count == 0


def test_thinking_mode_include_thoughts():
    from patterpunk.llm.thinking import ThinkingConfig

    model = GoogleModel(
        model="gemini-2.5-flash",
        location="us-central1",
        temperature=0.1,
        thinking_config=ThinkingConfig(token_budget=512, include_thoughts=True),
    )

    response = (
        Chat(model=model)
        .add_message(SystemMessage("You are a helpful assistant."))
        .add_message(UserMessage("Calculate the area of a circle with radius 5."))
        .complete()
        .latest_message
    )

    assert response.content is not None
    assert len(response.content) > 0
    content = response.content.lower()
    assert "circle" in content or "area" in content or "5" in content
    assert (
        response.thinking_token_count is not None and response.thinking_token_count > 0
    )
    assert response.has_thinking
    assert len(response.thinking_blocks) > 0


def test_thinking_mode_exclude_thoughts():
    from patterpunk.llm.thinking import ThinkingConfig

    model = GoogleModel(
        model="gemini-2.5-flash",
        location="us-central1",
        temperature=0.1,
        thinking_config=ThinkingConfig(token_budget=512, include_thoughts=False),
    )

    response = (
        Chat(model=model)
        .add_message(SystemMessage("You are a helpful assistant."))
        .add_message(UserMessage("What is 2 + 2?"))
        .complete()
        .latest_message
    )

    assert response.content is not None
    assert len(response.content) > 0
    content = response.content.lower()
    assert "4" in content or "four" in content
    assert (
        response.thinking_token_count is not None and response.thinking_token_count > 0
    )
    assert len(response.thinking_blocks) == 0


def test_thinking_token_count_reported():
    from patterpunk.llm.thinking import ThinkingConfig

    model = GoogleModel(
        model="gemini-2.5-flash",
        location="us-central1",
        temperature=0.1,
        thinking_config=ThinkingConfig(effort="medium", include_thoughts=True),
    )

    response = (
        Chat(model=model)
        .add_message(SystemMessage("You are a helpful assistant."))
        .add_message(UserMessage("Explain why the sky is blue in one sentence."))
        .complete()
        .latest_message
    )

    assert (
        response.thinking_token_count is not None
    ), "thinking_token_count should be reported when thinking is enabled"
    assert (
        response.thinking_token_count > 0
    ), "thinking_token_count should be positive when thinking is enabled"
    assert (
        response.has_thinking
    ), "has_thinking should be True when include_thoughts is enabled"
    assert (
        len(response.thinking_blocks) > 0
    ), "thinking_blocks should contain entries when include_thoughts is enabled"


def test_thinking_mode_deepcopy():
    import copy
    from patterpunk.llm.thinking import ThinkingConfig

    thinking_config = ThinkingConfig(token_budget=1024, include_thoughts=True)
    original_model = GoogleModel(
        model="gemini-2.5-flash",
        location="us-central1",
        thinking_config=thinking_config,
    )

    copied_model = copy.deepcopy(original_model)

    assert copied_model.thinking_config == original_model.thinking_config
    assert copied_model.thinking_budget == original_model.thinking_budget
    assert copied_model.include_thoughts == original_model.include_thoughts
    assert copied_model.model == original_model.model
    assert copied_model.location == original_model.location


def test_multimodal_image():
    model = GoogleModel(
        model="gemini-2.5-flash", location="us-central1", temperature=0.1
    )

    chat = Chat(model=model)

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
    model = GoogleModel(
        model="gemini-2.5-flash", location="us-central1", temperature=0.0
    )

    chat = Chat(model=model)

    title = (
        chat.add_message(
            SystemMessage(
                """Create a single-line title for the given document. It needs to be descriptive and short, and not copied from the document"""
            )
        )
        .add_message(
            UserMessage(
                content=[
                    TextChunk(
                        "Google requires at least one text block in a multimodal request"
                    ),
                    MultimodalChunk.from_file(get_resource("research.pdf")),
                ]
            )
        )
        .complete()
        .latest_message.content
    )

    assert "bank of canada" in title.lower()
    # The model should mention research or economic content
    assert "research" in title.lower() or "economic" in title.lower()


def test_diagnostics_and_safety_filter_integration():
    """Live integration: confirms disable_safety_filters reaches Vertex without
    errors, finish_reason normalizes to STOP on a benign completion, and
    _provider.raw_finish_reason carries the native Vertex enum name."""
    model = GoogleModel(
        model="gemini-2.5-flash", location="us-central1", temperature=0.1
    )

    response = (
        Chat(model=model, disable_safety_filters=True)
        .add_message(UserMessage("Say hello in exactly one short sentence."))
        .complete()
        .latest_message
    )

    assert response.content
    assert len(response.content) > 0
    assert response.finish_reason == FinishReason.STOP
    assert response._provider.raw_finish_reason == "STOP"


# =============================================================================
# Gemini 3.5+ sampling-parameter deprecation gate
# =============================================================================


def make_stub_model(**kwargs):
    return GoogleModel(client=object(), **kwargs)


def test_gemini_25_keeps_sampling_params(caplog):
    model = make_stub_model(
        model="gemini-2.5-pro", temperature=0.3, top_p=0.8, top_k=20
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = model._build_generation_config(None, None, None, None)
    assert config.temperature == 0.3
    assert config.top_p == 0.8
    assert config.top_k == 20
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_gemini_36_omits_sampling_with_warning(caplog):
    model = make_stub_model(
        model="gemini-3.6-pro", temperature=0.3, top_p=0.8, top_k=20
    )
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = model._build_generation_config(None, None, None, None)
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "[GOOGLE]" in m
        and "temperature=0.3" in m
        and "top_p=0.8" in m
        and "top_k=20" in m
        for m in warning_messages
    )


def test_gemini_35_boundary_omits_sampling():
    model = make_stub_model(model="gemini-3.5-flash", temperature=0.3)
    config = model._build_generation_config(None, None, None, None)
    assert config.temperature is None


def test_gemini_36_at_defaults_is_silent(caplog):
    model = make_stub_model(model="gemini-3.6-flash")
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = model._build_generation_config(None, None, None, None)
    assert config.temperature is None
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_unknown_gemini_id_gets_newest_family_rules(caplog):
    model = make_stub_model(model="gemini-next", temperature=0.3)
    with caplog.at_level("WARNING", logger="patterpunk"):
        config = model._build_generation_config(None, None, None, None)
    assert config.temperature is None
    assert any("temperature=0.3" in r.message for r in caplog.records)
