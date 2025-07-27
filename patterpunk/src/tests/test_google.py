from typing import Optional, List

from pydantic import BaseModel, Field

from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import SystemMessage, UserMessage, ToolCallMessage
from patterpunk.llm.models.google import GoogleModel


def test_basic():
    model = GoogleModel(
        model="gemini-1.5-pro-002", location="northamerica-northeast1", temperature=1.1
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
        model="gemini-1.5-pro-002", location="northamerica-northeast1", temperature=0.1
    )
    # Create a chat instance with the parameterized model
    chat = Chat(model=model)

    # Sample text containing information about a book
    sample_text = """
    "To Kill a Mockingbird" is a novel by Harper Lee published in 1960. 
    It was immediately successful, winning the Pulitzer Prize, and has 
    become a classic of modern American literature. The plot and characters 
    are loosely based on Lee's observations of her family, her neighbors 
    and an event that occurred near her hometown of Monroeville, Alabama, in 1936, 
    when she was 10 years old.
    """

    # Add system message with instructions
    chat = chat.add_message(
        SystemMessage(
            """
            Extract information about the book from the provided text.
            Include the title, author, and publication year.
            If available, also extract the ISBN, genres, page count, and whether it's a bestseller.
            
            Strictly extract information about the book from the provided text and do not infer information.
            """
        )
    )

    # Add user message with the sample text and structured_output parameter
    chat = chat.add_message(
        UserMessage(sample_text, structured_output=ThoughtfulBookResponse)
    )

    # Complete the chat to get the response
    chat = chat.complete()

    # Access the parsed_output and verify the results
    parsed_output: ThoughtfulBookResponse = chat.parsed_output

    assert parsed_output.book_info.author == "Harper Lee"
    assert parsed_output.book_info.title == "To Kill a Mockingbird"
    assert parsed_output.book_info.genres is None
    assert parsed_output.book_info.publication_year == 1960


def test_available_models():
    us_models = GoogleModel.get_available_models(location="us-central1")
    assert "gemini-2.0-flash-001" in us_models
    assert len(us_models) > 3

    canada_models = GoogleModel.get_available_models(location="northamerica-northeast1")
    assert "gemini-1.5-pro-002" in canada_models
    assert len(canada_models) > 3


def test_tool_calling():
    """Test tool calling functionality with Google model."""
    from patterpunk.llm.chat import Chat
    
    def calculate_area(length: float, width: float) -> str:
        """Calculate the area of a rectangle given length and width."""
        area = length * width
        return f"The area is {area} square units"
    
    def get_math_fact(topic: str) -> str:
        """Get an interesting mathematical fact about a topic."""
        facts = {
            "rectangle": "A rectangle has opposite sides that are equal and parallel",
            "area": "Area measures the amount of space inside a 2D shape",
            "geometry": "Geometry is one of the oldest mathematical sciences"
        }
        return facts.get(topic.lower(), "Mathematics is the language of the universe")
    
    model = GoogleModel(
        model="gemini-1.5-pro-002", 
        location="northamerica-northeast1", 
        temperature=0.1
    )
    
    chat = Chat(model=model).with_tools([calculate_area, get_math_fact])
    
    response = (
        chat
        .add_message(SystemMessage("You are a geometry helper. Use tools to solve problems."))
        .add_message(UserMessage(
            "I have a rectangle that is 5 units long and 3 units wide. "
            "Calculate its area and give me an interesting fact about rectangles."
        ))
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


def test_simple_tool_calling():
    """Test simple tool calling with a single function."""
    from patterpunk.llm.chat import Chat
    
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        return f"The weather in {location} is sunny and 22°C"
    
    model = GoogleModel(
        model="gemini-1.5-pro-002", 
        location="northamerica-northeast1", 
        temperature=0.1
    )
    
    chat = Chat(model=model).with_tools([get_weather])
    
    response = (
        chat
        .add_message(UserMessage("What's the weather like in Paris?"))
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
        # Response should mention Paris and weather if not a tool call
        content = response.latest_message.content.lower()
        assert "paris" in content


def test_thinking_mode_fixed_budget():
    """Test thinking mode with a fixed budget."""
    from patterpunk.llm.thinking import ThinkingConfig
    model = GoogleModel(
        model="gemini-2.5-flash",
        location="northamerica-northeast1", 
        temperature=0.1,
        thinking_config=ThinkingConfig(token_budget=1024)
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


def test_thinking_mode_dynamic_budget():
    """Test thinking mode with dynamic budget (effort=high)."""
    from patterpunk.llm.thinking import ThinkingConfig
    model = GoogleModel(
        model="gemini-2.5-flash",
        location="northamerica-northeast1", 
        temperature=0.1,
        thinking_config=ThinkingConfig(effort="high")
    )
    
    response = (
        Chat(model=model)
        .add_message(SystemMessage("You are a helpful assistant."))
        .add_message(UserMessage("Explain the concept of recursion in programming with an example."))
        .complete()
        .latest_message
    )
    
    assert response.content is not None
    assert len(response.content) > 100
    content = response.content.lower()
    assert "recursion" in content


def test_thinking_mode_disabled():
    """Test thinking mode disabled (budget=0)."""
    from patterpunk.llm.thinking import ThinkingConfig
    model = GoogleModel(
        model="gemini-2.5-flash",
        location="northamerica-northeast1", 
        temperature=0.1,
        thinking_config=ThinkingConfig(token_budget=0)
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


def test_thinking_mode_include_thoughts():
    """Test thinking mode with thoughts included in response."""
    from patterpunk.llm.thinking import ThinkingConfig
    model = GoogleModel(
        model="gemini-2.5-flash",
        location="northamerica-northeast1", 
        temperature=0.1,
        thinking_config=ThinkingConfig(token_budget=512, include_thoughts=True)
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


def test_thinking_mode_exclude_thoughts():
    """Test thinking mode with thoughts excluded from response."""
    from patterpunk.llm.thinking import ThinkingConfig
    model = GoogleModel(
        model="gemini-2.5-flash",
        location="northamerica-northeast1", 
        temperature=0.1,
        thinking_config=ThinkingConfig(token_budget=512, include_thoughts=False)
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


def test_thinking_mode_deepcopy():
    """Test that thinking mode parameters are preserved in deepcopy."""
    import copy
    from patterpunk.llm.thinking import ThinkingConfig
    
    thinking_config = ThinkingConfig(token_budget=1024, include_thoughts=True)
    original_model = GoogleModel(
        model="gemini-2.5-flash",
        location="northamerica-northeast1",
        thinking_config=thinking_config
    )
    
    copied_model = copy.deepcopy(original_model)
    
    assert copied_model.thinking_config == original_model.thinking_config
    assert copied_model.thinking_budget == original_model.thinking_budget
    assert copied_model.include_thoughts == original_model.include_thoughts
    assert copied_model.model == original_model.model
    assert copied_model.location == original_model.location
