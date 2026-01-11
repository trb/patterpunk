import pytest
from pydantic import BaseModel, Field
from typing import List, Optional
from patterpunk.llm.chat.core import Chat
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.messages.system import SystemMessage
from patterpunk.llm.messages.tool_call import ToolCallMessage
from patterpunk.llm.messages.user import UserMessage
from patterpunk.llm.chunks import CacheChunk, MultimodalChunk
from tests.test_utils import get_resource


def test_basic():
    print()
    print()
    print("available models")
    print(AnthropicModel.get_available_models())
    print()

    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001", max_tokens=4096, temperature=0.1
        )
    )

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

    print(chat.latest_message.content)
    print()
    print(chat.extract_json())


def test_structured_output():
    """
    @todo Add an assert that checks that fields that can't be answered from the text are actually set to none. This
    ensures the models don't hallucinate.
    """

    class Author(BaseModel):
        name: str
        expertise: List[str]
        years_experience: Optional[int] = None

    class Reference(BaseModel):
        title: str
        year: int
        url: Optional[str] = None

    class DocumentAnalysis(BaseModel):
        title: str
        summary: str
        key_points: List[str]
        sentiment: str = Field(
            description="Overall sentiment of the document (positive, negative, neutral)"
        )
        topics: List[str] = Field(min_length=3, max_length=10)
        author: Author
        references: List[Reference] = Field(default_factory=list)
        confidence_score: float = Field(ge=0.0, le=1.0)
        is_factual: bool

    sonnet_chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001", max_tokens=4096, temperature=0.2
        )
    )

    sonnet_chat = (
        sonnet_chat.add_message(
            SystemMessage(
                """
You are an expert document analyzer. Your task is to analyze the provided text and extract structured information.
Be thorough and accurate in your analysis.
"""
            )
        )
        .add_message(
            UserMessage(
                """
Please analyze this research paper abstract:

Title: The Impact of Artificial Intelligence on Climate Change Mitigation

Abstract:
This paper examines the dual role of artificial intelligence (AI) in addressing climate change. 
We analyze how AI technologies can both contribute to and help mitigate greenhouse gas emissions. 
Our research indicates that while AI systems require significant energy for training and inference, 
they also enable more efficient energy management, improved climate modeling, and optimization of 
renewable energy systems. Through case studies across five continents, we demonstrate that 
AI-optimized systems can reduce energy consumption by 15-30% in various sectors. However, we also 
identify ethical concerns regarding the unequal distribution of both AI benefits and climate impacts. 
The paper concludes with policy recommendations for ensuring that AI development aligns with climate 
goals and promotes environmental justice.

Keywords: artificial intelligence, climate change, energy efficiency, environmental policy, ethics
""",
                structured_output=DocumentAnalysis,
            )
        )
        .complete()
    )

    sonnet_parsed = sonnet_chat.parsed_output

    assert isinstance(
        sonnet_parsed, DocumentAnalysis
    ), "Sonnet output should be a DocumentAnalysis instance"
    assert sonnet_parsed.title, "Title should not be empty"
    assert (
        "artificial intelligence" in sonnet_parsed.title.lower()
        or "ai" in sonnet_parsed.title.lower()
    ), "Title should mention AI"
    assert "climate" in sonnet_parsed.title.lower(), "Title should mention climate"

    assert len(sonnet_parsed.summary) > 50, "Summary should be substantial"
    assert len(sonnet_parsed.key_points) >= 3, "Should identify at least 3 key points"

    assert sonnet_parsed.sentiment in [
        "positive",
        "negative",
        "neutral",
        "mixed",
    ], "Sentiment should be a valid value"
    assert len(sonnet_parsed.topics) >= 3, "Should identify at least 3 topics"
    assert (
        "artificial intelligence" in " ".join(sonnet_parsed.topics).lower()
        or "ai" in " ".join(sonnet_parsed.topics).lower()
    ), "Topics should include AI"
    assert (
        "climate" in " ".join(sonnet_parsed.topics).lower()
    ), "Topics should include climate"

    assert sonnet_parsed.author.name, "Author name should not be empty"
    assert len(sonnet_parsed.author.expertise) > 0, "Author should have expertise"

    assert (
        0 <= sonnet_parsed.confidence_score <= 1
    ), "Confidence score should be between 0 and 1"
    assert isinstance(sonnet_parsed.is_factual, bool), "is_factual should be a boolean"

    assert (
        sonnet_parsed.author.years_experience is None
    ), "Author years_experience should be None as it's not in the text"
    for ref in sonnet_parsed.references:
        assert ref.url is None, "Reference URL should be None as it's not in the text"

    haiku_chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001", max_tokens=4096, temperature=0.2
        )
    )

    haiku_chat = (
        haiku_chat.add_message(
            SystemMessage(
                """
You are an expert document analyzer. Your task is to analyze the provided text and extract structured information.
Be thorough and accurate in your analysis.
"""
            )
        )
        .add_message(
            UserMessage(
                """
Please analyze this research paper abstract:

Title: The Impact of Artificial Intelligence on Climate Change Mitigation

Abstract:
This paper examines the dual role of artificial intelligence (AI) in addressing climate change. 
We analyze how AI technologies can both contribute to and help mitigate greenhouse gas emissions. 
Our research indicates that while AI systems require significant energy for training and inference, 
they also enable more efficient energy management, improved climate modeling, and optimization of 
renewable energy systems. Through case studies across five continents, we demonstrate that 
AI-optimized systems can reduce energy consumption by 15-30% in various sectors. However, we also 
identify ethical concerns regarding the unequal distribution of both AI benefits and climate impacts. 
The paper concludes with policy recommendations for ensuring that AI development aligns with climate 
goals and promotes environmental justice.

Keywords: artificial intelligence, climate change, energy efficiency, environmental policy, ethics
""",
                structured_output=DocumentAnalysis,
            )
        )
        .complete()
    )

    haiku_parsed = haiku_chat.parsed_output

    assert isinstance(
        haiku_parsed, DocumentAnalysis
    ), "Haiku output should be a DocumentAnalysis instance"
    assert haiku_parsed.title, "Title should not be empty"
    assert (
        "artificial intelligence" in haiku_parsed.title.lower()
        or "ai" in haiku_parsed.title.lower()
    ), "Title should mention AI"
    assert "climate" in haiku_parsed.title.lower(), "Title should mention climate"

    assert len(haiku_parsed.summary) > 50, "Summary should be substantial"
    assert len(haiku_parsed.key_points) >= 3, "Should identify at least 3 key points"

    assert haiku_parsed.sentiment in [
        "positive",
        "negative",
        "neutral",
        "mixed",
    ], "Sentiment should be a valid value"
    assert len(haiku_parsed.topics) >= 3, "Should identify at least 3 topics"
    assert (
        "artificial intelligence" in " ".join(haiku_parsed.topics).lower()
        or "ai" in " ".join(haiku_parsed.topics).lower()
    ), "Topics should include AI"
    assert (
        "climate" in " ".join(haiku_parsed.topics).lower()
    ), "Topics should include climate"

    assert haiku_parsed.author.name, "Author name should not be empty"
    assert len(haiku_parsed.author.expertise) > 0, "Author should have expertise"

    assert (
        0 <= haiku_parsed.confidence_score <= 1
    ), "Confidence score should be between 0 and 1"
    assert isinstance(haiku_parsed.is_factual, bool), "is_factual should be a boolean"

    assert (
        haiku_parsed.author.years_experience is None
    ), "Author years_experience should be None as it's not in the text"
    for ref in haiku_parsed.references:
        assert ref.url is None, "Reference URL should be None as it's not in the text"

    assert isinstance(sonnet_parsed.confidence_score, float) and isinstance(
        haiku_parsed.confidence_score, float
    ), "Both models should return confidence scores as floats"
    assert (
        len(sonnet_parsed.key_points) > 0 and len(haiku_parsed.key_points) > 0
    ), "Both models should identify key points"


def test_reasoning_mode_version_parsing():

    model_37 = AnthropicModel(model="claude-3-7-sonnet-20250219")
    assert model_37._parse_model_version() == (3, 7)
    assert model_37._is_reasoning_model() == True

    # Claude Haiku 4.5 - version 4.5, IS a reasoning model (>= 3.7)
    model_45_haiku = AnthropicModel(model="claude-haiku-4-5-20251001")
    assert model_45_haiku._parse_model_version() == (4, 5)
    assert model_45_haiku._is_reasoning_model() == True

    # Old Claude 3 Haiku - NOT a reasoning model (< 3.7)
    model_3_haiku = AnthropicModel(model="claude-3-haiku-20240307")
    assert model_3_haiku._parse_model_version() == (3, 0)
    assert model_3_haiku._is_reasoning_model() == False

    model_opus4 = AnthropicModel(model="claude-opus-4-20250514")
    assert model_opus4._parse_model_version() == (4, 0)
    assert model_opus4._is_reasoning_model() == True

    model_sonnet4 = AnthropicModel(model="claude-sonnet-4-20250514")
    assert model_sonnet4._parse_model_version() == (4, 0)
    assert model_sonnet4._is_reasoning_model() == True

    model_45 = AnthropicModel(model="claude-sonnet-4-5-20250614")
    assert model_45._parse_model_version() == (4, 5)
    assert model_45._is_reasoning_model() == True

    model_5 = AnthropicModel(model="claude-opus-5-20250714")
    assert model_5._parse_model_version() == (5, 0)
    assert model_5._is_reasoning_model() == True

    model_51 = AnthropicModel(model="claude-sonnet-5-1-20250814")
    assert model_51._parse_model_version() == (5, 1)
    assert model_51._is_reasoning_model() == True

    model_unknown = AnthropicModel(model="some-unknown-model")
    assert model_unknown._parse_model_version() == (0, 0)
    assert model_unknown._is_reasoning_model() == False


def test_reasoning_mode_parameter_compatibility():

    model = AnthropicModel(model="claude-3-7-sonnet-20250219")
    api_params = {
        "model": "claude-3-7-sonnet-20250219",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 1000,
    }

    filtered_params = model._get_compatible_params(api_params)
    assert filtered_params == api_params

    model_with_thinking = AnthropicModel(
        model="claude-3-7-sonnet-20250219",
        thinking_config=ThinkingConfig(token_budget=4000),
    )

    filtered_params = model_with_thinking._get_compatible_params(api_params)
    expected_params = {
        "model": "claude-3-7-sonnet-20250219",
        "temperature": 1.0,
        "max_tokens": 1000,
    }
    assert filtered_params == expected_params
    assert "top_p" not in filtered_params
    assert "top_k" not in filtered_params

    assert filtered_params["temperature"] == 1.0


def test_reasoning_mode_initialization():

    thinking_config = ThinkingConfig(token_budget=8000)

    model = AnthropicModel(
        model="claude-sonnet-4-20250514",
        thinking_config=thinking_config,
        temperature=0.5,
        max_tokens=2000,
    )

    assert model.thinking_config == thinking_config
    assert model.thinking.type == "enabled"
    assert model.thinking.budget_tokens == 8000
    assert model.model == "claude-sonnet-4-20250514"
    assert model.temperature == 0.5
    assert model.max_tokens == 2000
    assert model._is_reasoning_model() == True


def test_reasoning_mode_default_type():

    thinking_config = ThinkingConfig(token_budget=6000)

    model = AnthropicModel(
        model="claude-opus-4-20250514",
        thinking_config=thinking_config,
        temperature=0.3,
        max_tokens=1500,
    )

    assert model.thinking.type == "enabled"
    assert model.thinking.budget_tokens == 6000
    assert model._is_reasoning_model() == True

    api_params = {
        "model": "claude-opus-4-20250514",
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 30,
        "max_tokens": 1500,
    }

    filtered_params = model._get_compatible_params(api_params)

    assert "top_p" not in filtered_params
    assert "top_k" not in filtered_params
    assert filtered_params["temperature"] == 1.0


def test_reasoning_mode_with_claude_sonnet_4():
    from patterpunk.llm.chat.core import Chat

    thinking_config = ThinkingConfig(token_budget=4000)

    chat = Chat(
        model=AnthropicModel(
            model="claude-sonnet-4-20250514",
            thinking_config=thinking_config,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=2000,
        )
    )

    assert chat.model.thinking_config == thinking_config
    assert chat.model._is_reasoning_model() == True
    assert chat.model._parse_model_version() == (4, 0)

    api_params = {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 2000,
        "system": "You are a helpful assistant.",
        "messages": [],
    }

    filtered_params = chat.model._get_compatible_params(api_params)

    assert "thinking" not in filtered_params
    assert "top_p" not in filtered_params
    assert "top_k" not in filtered_params
    assert "temperature" in filtered_params
    assert filtered_params["temperature"] == 1.0
    assert filtered_params["max_tokens"] == 2000

    chat_37 = Chat(
        model=AnthropicModel(
            model="claude-3-7-sonnet-20250219",
            thinking_config=ThinkingConfig(token_budget=2000),
            temperature=0.5,
        )
    )

    assert chat_37.model._is_reasoning_model() == True
    assert chat_37.model._parse_model_version() == (3, 7)

    # Test non-reasoning model (Claude 3 Haiku, version < 3.7)
    chat_3_haiku = Chat(
        model=AnthropicModel(
            model="claude-3-haiku-20240307", temperature=0.5, top_p=0.8, top_k=50
        )
    )

    assert chat_3_haiku.model._is_reasoning_model() == False
    assert chat_3_haiku.model._parse_model_version() == (3, 0)

    non_reasoning_params = {
        "temperature": 0.5,
        "top_p": 0.8,
        "top_k": 50,
        "max_tokens": 1000,
    }
    filtered_non_reasoning = chat_3_haiku.model._get_compatible_params(
        non_reasoning_params
    )
    assert filtered_non_reasoning == non_reasoning_params


def test_reasoning_mode_plain_text_response():
    from patterpunk.llm.chat.core import Chat

    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=2000),
            max_tokens=4000,
            temperature=1.0,
        )
    )

    response = (
        chat.add_message(SystemMessage("You are a helpful math tutor."))
        .add_message(
            UserMessage("Explain why 2 + 2 = 4 using basic mathematical principles.")
        )
        .complete()
    )

    assert response.latest_message is not None
    assert response.latest_message.content is not None
    assert len(response.latest_message.content.strip()) > 0

    content_lower = response.latest_message.content.lower()
    assert any(
        term in content_lower for term in ["addition", "sum", "math", "number", "equal"]
    )


def test_reasoning_mode_structured_output():
    from patterpunk.llm.chat.core import Chat

    class MathExplanation(BaseModel):
        concept: str = Field(description="The mathematical concept being explained")
        explanation: str = Field(description="Clear explanation of the concept")
        examples: List[str] = Field(description="2-3 simple examples")
        difficulty_level: str = Field(description="beginner, intermediate, or advanced")
        confidence: float = Field(
            ge=0.0, le=1.0, description="Confidence in explanation accuracy"
        )

    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=3000),
            max_tokens=5000,
            temperature=1.0,
        )
    )

    response = (
        chat.add_message(
            SystemMessage("You are a math educator. Provide structured explanations.")
        )
        .add_message(
            UserMessage(
                "Explain the concept of multiplication.",
                structured_output=MathExplanation,
            )
        )
        .complete()
    )

    assert response.parsed_output is not None
    assert isinstance(response.parsed_output, MathExplanation)

    parsed = response.parsed_output
    assert parsed.concept
    assert "multiplication" in parsed.concept.lower()
    assert parsed.explanation
    assert len(parsed.examples) >= 2
    assert parsed.difficulty_level in ["beginner", "intermediate", "advanced"]
    assert 0.0 <= parsed.confidence <= 1.0


def test_reasoning_mode_tool_calling():
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

    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=4000),
            max_tokens=6000,
            temperature=1.0,
        )
    ).with_tools([calculate_area, get_math_fact])

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


def test_multimodal_image():
    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001", temperature=0.1, max_tokens=4096
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
    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001", temperature=0.0, max_tokens=4096
        )
    )

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
    """Test simple tool calling with Anthropic"""

    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city or location to get weather for
        """
        return f"The weather in {location} is sunny and 22°C"

    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001", temperature=0.0, max_tokens=4096
        )
    ).with_tools([get_weather])

    system_msg = SystemMessage(
        "You are a helpful assistant that MUST use the provided tools to answer questions. "
        "When asked about weather, you MUST call the get_weather tool. "
        "Do not just describe what you would do - actually call the tool."
    )

    # Use execute_tools=False to inspect the ToolCallMessage before execution
    response = (
        chat.add_message(system_msg)
        .add_message(UserMessage("What's the weather in Paris?"))
        .complete(execute_tools=False)
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
    assert tool_call.type == "function"
    assert tool_call.name == "get_weather"

    import json

    arguments = json.loads(tool_call.arguments)
    assert "location" in arguments
    assert "paris" in arguments["location"].lower()


def test_multi_tool_calling():
    """Test calling multiple tools with Anthropic"""

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
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001", temperature=0.0, max_tokens=4096
        )
    ).with_tools([calculate_area, get_math_fact])

    system_msg = SystemMessage(
        "You are a geometry helper that MUST use the provided tools to solve problems. "
        "When asked to calculate area, you MUST call the calculate_area tool. "
        "When asked for facts, you MUST call the get_math_fact tool. "
        "Do not calculate or provide facts without using the tools."
    )

    # Use execute_tools=False to inspect the ToolCallMessage before execution
    response = (
        chat.add_message(system_msg)
        .add_message(
            UserMessage(
                "I have a rectangle that is 5 units long and 3 units wide. "
                "Calculate its area and give me an interesting fact about rectangles."
            )
        )
        .complete(execute_tools=False)
    )

    assert response.latest_message is not None
    assert isinstance(
        response.latest_message, ToolCallMessage
    ), f"Expected ToolCallMessage but got {type(response.latest_message).__name__}. Content: {response.latest_message.content}"

    tool_calls = response.latest_message.tool_calls
    assert (
        len(tool_calls) >= 1
    ), f"Expected at least one tool call, got {len(tool_calls)}"

    # Verify we have the expected tool calls (using dataclass attribute access)
    tool_names = [tc.name for tc in tool_calls]
    assert (
        "calculate_area" in tool_names or "get_math_fact" in tool_names
    ), f"Expected calculate_area or get_math_fact in tool calls, got: {tool_names}"


def test_cache_chunks():
    """Test that cache chunks work with Anthropic"""

    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001", temperature=0.1, max_tokens=4096
        )
    )

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


def test_thinking_blocks_preservation():
    """Test that thinking blocks are extracted and preserved from API responses"""
    from patterpunk.llm.chat.core import Chat
    from patterpunk.llm.messages.assistant import AssistantMessage

    # Test with Claude 4.5 Sonnet (reasoning model)
    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=2000),
            max_tokens=4000,
            temperature=1.0,
        )
    )

    response = (
        chat.add_message(
            SystemMessage(
                "You are a helpful assistant. Think through problems step by step."
            )
        )
        .add_message(
            UserMessage("What is 27 * 453? Think through the calculation step by step.")
        )
        .complete()
    )

    # Verify the response has thinking blocks
    assert response.latest_message is not None
    assert isinstance(response.latest_message, AssistantMessage)
    assert hasattr(response.latest_message, "thinking_blocks")

    # Claude 4+ models return summarized thinking, but should still have thinking blocks
    if response.latest_message.has_thinking:
        assert len(response.latest_message.thinking_blocks) > 0
        # Verify thinking blocks have the correct structure
        for block in response.latest_message.thinking_blocks:
            assert "type" in block
            assert block["type"] in ["thinking", "redacted_thinking"]
            if block["type"] == "thinking":
                assert "thinking" in block
                # Signature may or may not be present depending on model
            elif block["type"] == "redacted_thinking":
                assert "data" in block

    # Verify the final answer content is present
    assert response.latest_message.content is not None
    assert len(response.latest_message.content.strip()) > 0


def test_thinking_blocks_with_tool_calling():
    """Test that thinking blocks are preserved in multi-turn conversations with tool use"""
    from patterpunk.llm.chat.core import Chat

    def multiply(a: float, b: float) -> str:
        """Multiply two numbers together.

        Args:
            a: First number
            b: Second number
        """
        return str(a * b)

    chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=3000),
            max_tokens=4000,
            temperature=1.0,
        )
    ).with_tools([multiply])

    # Make initial request that should use tools
    response = (
        chat.add_message(
            SystemMessage(
                "You are a math assistant. Use the multiply tool when asked to multiply numbers."
            )
        )
        .add_message(UserMessage("What is 27 times 453? Use the multiply tool."))
        .complete()
    )

    # Check if we got a tool call
    if isinstance(response.latest_message, ToolCallMessage):
        # Verify thinking blocks are present on the tool call message
        assert hasattr(response.latest_message, "thinking_blocks")

        # The thinking blocks should be preserved when converting back to Anthropic format
        # This is critical for multi-turn conversations
        messages_for_api = response.model._convert_messages_for_anthropic(
            response.messages
        )

        # Find the assistant message with tool_use
        tool_use_message = None
        for msg in messages_for_api:
            if msg["role"] == "assistant" and any(
                block.get("type") == "tool_use" for block in msg["content"]
            ):
                tool_use_message = msg
                break

        if tool_use_message and response.latest_message.thinking_blocks:
            # Verify thinking blocks come before tool_use blocks
            content_blocks = tool_use_message["content"]
            thinking_indices = [
                i
                for i, block in enumerate(content_blocks)
                if block.get("type") in ["thinking", "redacted_thinking"]
            ]
            tool_use_indices = [
                i
                for i, block in enumerate(content_blocks)
                if block.get("type") == "tool_use"
            ]

            # If both exist, thinking should come before tool_use
            if thinking_indices and tool_use_indices:
                assert max(thinking_indices) < min(
                    tool_use_indices
                ), "Thinking blocks must come before tool_use blocks in Anthropic API format"


def test_chat_persistence_and_resumption():
    """
    Test that chat history can be extracted, stored (simulated), and resumed.

    This validates the persistence story for patterpunk:
    1. Create a chat with tools and reasoning mode
    2. Have the model call a tool with a specific input
    3. Extract all message data and create NEW message objects (simulating DB load)
    4. Resume the chat and verify the model can reference earlier tool call data

    The key validation is that the model in step 4 correctly references:
    - The input argument from the ToolCallMessage
    - The output from the ToolResultMessage
    """
    from patterpunk.llm.messages.assistant import AssistantMessage
    from patterpunk.llm.messages.tool_result import ToolResultMessage
    from patterpunk.llm.types import ToolCall

    # --- Setup: Define a tool with predictable input/output ---
    def store_secret(input_code: str) -> str:
        """Store a secret code and return a confirmation code.

        Args:
            input_code: The secret code to store
        """
        # Return a predictable but different output
        return f"STORED-{input_code}-CONFIRMED-8472"

    # --- Phase 1: Create initial chat and make a tool call ---
    original_chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=2000),
            max_tokens=4000,
            temperature=1.0,
        )
    ).with_tools([store_secret])

    system_msg = SystemMessage(
        "You are a helpful assistant. When asked to store a secret code, "
        "you MUST use the store_secret tool with the exact code provided."
    )

    # Use a unique input code that we can verify later
    user_msg = UserMessage(
        "Please store the secret code ALPHA-9999 using the store_secret tool."
    )

    # Complete the chat with automatic tool execution
    completed_chat = (
        original_chat.add_message(system_msg).add_message(user_msg).complete()
    )

    # Verify we got a final response (tool was executed)
    assert completed_chat.latest_message is not None
    assert isinstance(
        completed_chat.latest_message, AssistantMessage
    ), f"Expected final AssistantMessage but got {type(completed_chat.latest_message).__name__}"

    # --- Phase 2: Extract and reconstruct all messages ---
    # This simulates loading from a database

    reconstructed_messages = []

    for msg in completed_chat.messages:
        if isinstance(msg, SystemMessage):
            # Reconstruct SystemMessage
            new_msg = SystemMessage(msg.content)
            reconstructed_messages.append(new_msg)

        elif isinstance(msg, UserMessage):
            # Reconstruct UserMessage
            new_msg = UserMessage(msg.content)
            reconstructed_messages.append(new_msg)

        elif isinstance(msg, ToolCallMessage):
            # Reconstruct ToolCallMessage with new ToolCall dataclasses
            new_tool_calls = []
            for tc in msg.tool_calls:
                new_tc = ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    type=tc.type,
                )
                new_tool_calls.append(new_tc)

            # Also reconstruct thinking_blocks if present
            new_thinking_blocks = None
            if msg.thinking_blocks:
                # Deep copy the thinking blocks (simulating JSON deserialization)
                import json

                new_thinking_blocks = json.loads(json.dumps(msg.thinking_blocks))

            new_msg = ToolCallMessage(
                new_tool_calls, thinking_blocks=new_thinking_blocks
            )
            reconstructed_messages.append(new_msg)

        elif isinstance(msg, ToolResultMessage):
            # Reconstruct ToolResultMessage
            new_msg = ToolResultMessage(
                content=msg.content,
                call_id=msg.call_id,
                function_name=msg.function_name,
                is_error=msg.is_error,
            )
            reconstructed_messages.append(new_msg)

        elif isinstance(msg, AssistantMessage):
            # Reconstruct AssistantMessage with thinking_blocks if present
            new_thinking_blocks = None
            if msg.thinking_blocks:
                import json

                new_thinking_blocks = json.loads(json.dumps(msg.thinking_blocks))

            new_msg = AssistantMessage(
                content=msg.content,
                thinking_blocks=new_thinking_blocks,
            )
            reconstructed_messages.append(new_msg)

    # --- Phase 3: Create a new chat with reconstructed messages ---
    # This simulates loading a chat from the database and resuming it

    resumed_chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=2000),
            max_tokens=4000,
            temperature=1.0,
        )
    ).with_tools([store_secret])

    # Add all reconstructed messages
    for msg in reconstructed_messages:
        resumed_chat = resumed_chat.add_message(msg)

    # --- Phase 4: Ask a follow-up question that requires referencing earlier data ---
    follow_up = UserMessage(
        "What was the exact input code you used when calling the store_secret tool, "
        "and what was the confirmation code you received back?"
    )

    final_chat = resumed_chat.add_message(follow_up).complete()

    # --- Phase 5: Verify the model correctly references the earlier data ---
    final_response = final_chat.latest_message
    assert final_response is not None
    assert isinstance(final_response, AssistantMessage)

    response_content = final_response.content.upper()

    # The model should mention the input code from the ToolCallMessage arguments
    assert (
        "ALPHA-9999" in response_content
    ), f"Model should reference input code ALPHA-9999. Got: {final_response.content}"

    # The model should mention the confirmation code from the ToolResultMessage
    assert (
        "8472" in response_content or "CONFIRMED" in response_content
    ), f"Model should reference confirmation code (8472 or CONFIRMED). Got: {final_response.content}"

    print("\n=== Chat Persistence Test Passed ===")
    print(f"Original messages: {len(completed_chat.messages)}")
    print(f"Reconstructed messages: {len(reconstructed_messages)}")
    print(
        f"Final response mentions both input (ALPHA-9999) and output (8472/CONFIRMED)"
    )


@pytest.mark.asyncio
async def test_chat_persistence_and_resumption_streaming():
    """
    Test that chat history from async streaming can be extracted, stored, and resumed.

    This is the streaming version of test_chat_persistence_and_resumption.
    It validates that the streaming API produces the same persistable chat state.

    The key validation is that after streaming:
    1. The chat contains all messages (including ToolCallMessage, ToolResultMessage)
    2. These can be reconstructed and used to resume the conversation
    3. The model correctly references data from the earlier tool interactions
    """
    from patterpunk.llm.messages.assistant import AssistantMessage
    from patterpunk.llm.messages.tool_result import ToolResultMessage
    from patterpunk.llm.types import ToolCall

    # --- Setup: Define a tool with predictable input/output ---
    def store_secret(input_code: str) -> str:
        """Store a secret code and return a confirmation code.

        Args:
            input_code: The secret code to store
        """
        return f"STORED-{input_code}-CONFIRMED-8472"

    # --- Phase 1: Create initial chat and stream a tool call ---
    original_chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=2000),
            max_tokens=4000,
            temperature=1.0,
        )
    ).with_tools([store_secret])

    system_msg = SystemMessage(
        "You are a helpful assistant. When asked to store a secret code, "
        "you MUST use the store_secret tool with the exact code provided."
    )

    user_msg = UserMessage(
        "Please store the secret code BETA-7777 using the store_secret tool."
    )

    # Stream the completion (tool will be auto-executed)
    chat_with_messages = original_chat.add_message(system_msg).add_message(user_msg)

    phase1_iterations = 0
    async with chat_with_messages.complete_stream() as stream:
        async for _ in stream.content:
            phase1_iterations += 1

    # Get the final chat state after streaming
    completed_chat = await stream.chat

    # Verify we actually streamed (not a single-chunk response)
    assert phase1_iterations >= 3, (
        f"Expected at least 3 streaming iterations in phase 1, got {phase1_iterations}. "
        "This may indicate streaming is not working correctly."
    )

    # Verify we got a final response (tool was executed)
    assert completed_chat.latest_message is not None
    assert isinstance(
        completed_chat.latest_message, AssistantMessage
    ), f"Expected final AssistantMessage but got {type(completed_chat.latest_message).__name__}"

    # --- Phase 2: Extract and reconstruct all messages ---
    reconstructed_messages = []

    for msg in completed_chat.messages:
        if isinstance(msg, SystemMessage):
            new_msg = SystemMessage(msg.content)
            reconstructed_messages.append(new_msg)

        elif isinstance(msg, UserMessage):
            new_msg = UserMessage(msg.content)
            reconstructed_messages.append(new_msg)

        elif isinstance(msg, ToolCallMessage):
            new_tool_calls = []
            for tc in msg.tool_calls:
                new_tc = ToolCall(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    type=tc.type,
                )
                new_tool_calls.append(new_tc)

            new_thinking_blocks = None
            if msg.thinking_blocks:
                import json

                new_thinking_blocks = json.loads(json.dumps(msg.thinking_blocks))

            new_msg = ToolCallMessage(
                new_tool_calls, thinking_blocks=new_thinking_blocks
            )
            reconstructed_messages.append(new_msg)

        elif isinstance(msg, ToolResultMessage):
            new_msg = ToolResultMessage(
                content=msg.content,
                call_id=msg.call_id,
                function_name=msg.function_name,
                is_error=msg.is_error,
            )
            reconstructed_messages.append(new_msg)

        elif isinstance(msg, AssistantMessage):
            new_thinking_blocks = None
            if msg.thinking_blocks:
                import json

                new_thinking_blocks = json.loads(json.dumps(msg.thinking_blocks))

            new_msg = AssistantMessage(
                content=msg.content,
                thinking_blocks=new_thinking_blocks,
            )
            reconstructed_messages.append(new_msg)

    # --- Phase 3: Create a new chat with reconstructed messages ---
    resumed_chat = Chat(
        model=AnthropicModel(
            model="claude-haiku-4-5-20251001",
            thinking_config=ThinkingConfig(token_budget=2000),
            max_tokens=4000,
            temperature=1.0,
        )
    ).with_tools([store_secret])

    for msg in reconstructed_messages:
        resumed_chat = resumed_chat.add_message(msg)

    # --- Phase 4: Stream a follow-up question ---
    follow_up = UserMessage(
        "What was the exact input code you used when calling the store_secret tool, "
        "and what was the confirmation code you received back?"
    )

    resumed_with_followup = resumed_chat.add_message(follow_up)

    phase4_iterations = 0
    async with resumed_with_followup.complete_stream() as stream:
        async for _ in stream.content:
            phase4_iterations += 1

    final_chat = await stream.chat

    # Verify we actually streamed on the resumed chat too
    assert phase4_iterations >= 3, (
        f"Expected at least 3 streaming iterations in phase 4, got {phase4_iterations}. "
        "This may indicate streaming is not working correctly on resumed chat."
    )

    # --- Phase 5: Verify the model correctly references the earlier data ---
    final_response = final_chat.latest_message
    assert final_response is not None
    assert isinstance(final_response, AssistantMessage)

    response_content = final_response.content.upper()

    # The model should mention the input code from the ToolCallMessage arguments
    assert (
        "BETA-7777" in response_content
    ), f"Model should reference input code BETA-7777. Got: {final_response.content}"

    # The model should mention the confirmation code from the ToolResultMessage
    assert (
        "8472" in response_content or "CONFIRMED" in response_content
    ), f"Model should reference confirmation code (8472 or CONFIRMED). Got: {final_response.content}"

    print("\n=== Streaming Chat Persistence Test Passed ===")
    print(f"Phase 1 streaming iterations: {phase1_iterations}")
    print(f"Phase 4 streaming iterations: {phase4_iterations}")
    print(f"Original messages: {len(completed_chat.messages)}")
    print(f"Reconstructed messages: {len(reconstructed_messages)}")
    print(f"Final response mentions both input (BETA-7777) and output (8472/CONFIRMED)")
