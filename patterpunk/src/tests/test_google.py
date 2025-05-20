from typing import Optional, List

from pydantic import BaseModel, Field

from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import SystemMessage, UserMessage
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
