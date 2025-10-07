from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from patterpunk.llm.cache import CacheChunk
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.multimodal import MultimodalChunk
from patterpunk.llm.thinking import ThinkingConfig
from tests.test_utils import get_resource


def test_basic():
    print()
    print("available openai models")
    print(OpenAiModel.get_available_models())
    print()

    chat = Chat(model=OpenAiModel(model="gpt-4o", temperature=0.1))

    chat = (
        chat.add_message(
            SystemMessage(
                """
Extract the most applicable date from the document based on the type of document you're dealing
with. Then write a title for the document, target about 6 words. Be extremely concise and information
dense.

Work step by step and show your reasoning. Seeing your inner thoughts is more important than getting
a final answer. Show your process.

Response with the following JSON structure:

```json
{"date": "date you picked", "title": "title you wrote"}
```
    """
            )
        )
        .add_message(
            UserMessage(
                """
Here's the beginning of the document:

===START_OF_BEGINNING===
Scanned Document

 —  @Busamitab

 Certificate No: 13145

 CERTIFICATE
 OF TECHNICAL
 COMPETENCE

 This Certificate confirms that

 Ian Bailey

 Has demonstrated the standard of technical competence required for the
 management of a facility of the type set out below

 Facility Type

 Level 4 in Waste Management Operations -

 Managing Treatment Hazardous Waste (4TMH)

 Authorising Signatures:

 Chief Executive Officer £

 Director: PA

 Date of issue: 29 October 2013

 > «
 say

 CZ wamitab

 Qualification Title:

 WAMITAB Level 4 Diploma in Waste Management Operations : Managing
 Physical & Chemical Treatment - Hazardous Waste (QCF) - 4MPTH

 Qualification Accreditation Number:
 600/0331/5

 This Certificate is awarded to

 lan Bailey

 Awarded: 29/10/2013 Serial No:18862/4MPTH/1
 Authorised
 Ray Burberry

 Qualifications Manager, WAMITAB

 Units

 Y6021501
 H6021646
 J6021672

 K6009711

 M6009712
 A6021670
 K6021423
 M6021424
 D6021435
 K6021504
 U1051769
 F6021606

 CZ wamitabo

 Credit certificate
 This certificate determines credit awarded to:

 lan Bailey

 gained:

 Control work activities on a waste management facility

 Credit Credit
 Value Level

 Manage site operations for the treatment of hazardous waste 22 4
 Manage the transfer of outputs and disposal of residues from hazardous waste 13 4

 treatment and recovery operations

 Manage physical resources

 Manage the environmental impact of work actvities
 Manage the movement,
===END_OF_BEGINNING===

===START_OF_END===
nager, WAMITAB

 Regulated by

 Ofqual

 For more Information see http://register.ofqual.gov.uk

 The qualifications regulators logos on this certificate
 indicate that the qualification is accredited only for

 Serial No.: 18862/WM12/1

 HY FU
 a — LAY

 Llywodraeth Cymru Cymru
 Welsh Government

 ( 4 wamitabo

 Operator Competence Certificate

 Qualification Title:

 Managing Physical & Chemical Treatment - Hazardous Waste - 4MPTH

 This Certificate is awarded to

 lan Bailey

 Awarded: 29/10/2013

 Authorised

 ee TOE
 WAMITAB Chief Executive Officer CIWM Chief Executive Officer

 This certificate is jointly awarded by WAMITAB and the
 Chartered Institution of Wastes Management (CIWM)
 and provides evidence to meet the Operator
 Competence requirements of the Environmental
 Permitting (EP) Regulations, which came into force on
 6 April 2008.

 The Chartered Institution

 ey wamitalb

 Continuing Competence Certificate

 This certificate confirms that

 lan Bailey

 Has met the relevant requirements of the Continuing Competence scheme for the
 following award(s) which will remain current for two years from 28/02/2020

 LH Landfill - Hazardous Waste

 LIN Landfill - Inert Waste

 TMH Treatment - Hazardous Waste
 Verification date: 26/02/2020 Learner ID: 18862
 Authorised: Certificate No.: 5161522

 Date of Issue: 28/02/2020

 WAMITAB Chief Executive Officer CIWM Chief Executive Officer

 The Chartered Institution
 of Wastes Management

 00146135
===END_OF_END===
    """
            )
        )
        .complete()
    )

    print()
    print("message")
    print(chat.latest_message.content)
    print()
    print()
    print("json")
    print(chat.extract_json())


def test_o1():
    from patterpunk.llm.thinking import ThinkingConfig

    chat = Chat(
        model=OpenAiModel(
            model="o3-mini",
            temperature=0.1,
            thinking_config=ThinkingConfig(effort="medium"),
        )
    )

    chat = (
        chat.add_message(SystemMessage("""Only reply with 'test'"""))
        .add_message(UserMessage("You are being tested"))
        .complete()
    )

    print(chat.latest_message.content)


@pytest.mark.parametrize("model_name", ["gpt-4o", "o3-mini", "gpt-3.5-turbo"])
def test_structured_output(model_name):
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
            description="Think out loud about how you will complete the request. Be careful to catch edge cases and subtleties."
        )
        book_info: BookInfo = Field(
            description="The BookInfo structure representing the requested data. Follow the provided schema carefully. Do not infer fields, only include information that is present in the source message"
        )

    chat = Chat(model=OpenAiModel(model=model_name, temperature=0.1))

    sample_text = """
    "To Kill a Mockingbird" is a novel by Harper Lee published in 1960. 
    It was immediately successful, winning the Pulitzer Prize, and has 
    become a classic of modern American literature. The plot and characters 
    are loosely based on Lee's observations of her family, her neighbors 
    and an event that occurred near her hometown of Monroeville, Alabama, in 1936, 
    when she was 10 years old.
    """

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

    chat = chat.add_message(
        UserMessage(sample_text, structured_output=ThoughtfulBookResponse)
    )

    chat = chat.complete()

    parsed_output = chat.parsed_output.book_info

    assert parsed_output is not None
    assert isinstance(parsed_output, BookInfo)

    assert parsed_output.title == "To Kill a Mockingbird"
    assert parsed_output.author == "Harper Lee"
    assert parsed_output.publication_year == 1960

    assert parsed_output.isbn is None
    assert parsed_output.page_count is None
    assert parsed_output.is_bestseller is None

    assert parsed_output.genres is None or isinstance(parsed_output.genres, list)


def test_multimodal_image():
    chat = Chat(
        model=OpenAiModel(
            model="gpt-4.1-nano",
            temperature=0.1,
        )
    )

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
    chat = Chat(model=OpenAiModel(model="gpt-4.1-nano", temperature=0.0))

    title = (
        chat.add_message(
            SystemMessage(
                """Create a single-line title for the given document. It needs to be descriptive and short, and not copied from the document"""
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

    assert "bank of canada" in title.lower()
    assert "research" in title.lower()
    assert "2025" in title.lower()


def test_simple_tool_calling():
    """Test simple tool calling with OpenAI"""
    from patterpunk.llm.messages.tool_call import ToolCallMessage

    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city or location to get weather for
        """
        return f"The weather in {location} is sunny and 22°C"

    chat = Chat(model=OpenAiModel(model="gpt-4o-mini", temperature=0.0)).with_tools(
        [get_weather]
    )

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
    assert isinstance(
        response.latest_message, ToolCallMessage
    ), f"Expected ToolCallMessage but got {type(response.latest_message).__name__}. Content: {response.latest_message.content}"

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
    """Test calling multiple tools with OpenAI"""
    from patterpunk.llm.messages.tool_call import ToolCallMessage

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

    chat = Chat(model=OpenAiModel(model="gpt-4o-mini", temperature=0.0)).with_tools(
        [calculate_area, get_math_fact]
    )

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
    assert isinstance(
        response.latest_message, ToolCallMessage
    ), f"Expected ToolCallMessage but got {type(response.latest_message).__name__}. Content: {response.latest_message.content}"

    tool_calls = response.latest_message.tool_calls
    assert (
        len(tool_calls) >= 1
    ), f"Expected at least one tool call, got {len(tool_calls)}"

    # Verify we have the expected tool calls
    tool_names = [tc["function"]["name"] for tc in tool_calls]
    assert (
        "calculate_area" in tool_names or "get_math_fact" in tool_names
    ), f"Expected calculate_area or get_math_fact in tool calls, got: {tool_names}"
