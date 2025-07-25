from pydantic import BaseModel, Field
from typing import List, Optional
from patterpunk.llm.chat import Chat
from patterpunk.llm.models.anthropic import AnthropicModel, ThinkingConfig
from patterpunk.llm.messages import SystemMessage, UserMessage, ToolCallMessage


def test_basic():
    print()
    print()
    print("available models")
    print(AnthropicModel.get_available_models())
    print()

    chat = Chat(
        model=AnthropicModel(
            model="claude-3-5-sonnet-20240620", max_tokens=4096, temperature=0.1
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

    # Define a complex structured output type using Pydantic
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

    # Test with claude-3-7-sonnet-latest
    sonnet_chat = Chat(
        model=AnthropicModel(
            model="claude-3-7-sonnet-latest", max_tokens=4096, temperature=0.2
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

    # Validate sonnet model output
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

    # Check that fields that can't be answered from the text are set to None
    assert (
        sonnet_parsed.author.years_experience is None
    ), "Author years_experience should be None as it's not in the text"
    for ref in sonnet_parsed.references:
        assert ref.url is None, "Reference URL should be None as it's not in the text"

    # Test with claude-3-5-haiku-latest
    haiku_chat = Chat(
        model=AnthropicModel(
            model="claude-3-5-haiku-latest", max_tokens=4096, temperature=0.2
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

    # Validate haiku model output
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

    # Check that fields that can't be answered from the text are set to None
    assert (
        haiku_parsed.author.years_experience is None
    ), "Author years_experience should be None as it's not in the text"
    for ref in haiku_parsed.references:
        assert ref.url is None, "Reference URL should be None as it's not in the text"

    # Compare models
    assert isinstance(sonnet_parsed.confidence_score, float) and isinstance(
        haiku_parsed.confidence_score, float
    ), "Both models should return confidence scores as floats"
    assert (
        len(sonnet_parsed.key_points) > 0 and len(haiku_parsed.key_points) > 0
    ), "Both models should identify key points"


def test_reasoning_mode_version_parsing():
    """Test that version parsing works correctly for various model names."""
    
    # Test Claude 3.7 format
    model_37 = AnthropicModel(model="claude-3-7-sonnet-20250219")
    assert model_37._parse_model_version() == (3, 7)
    assert model_37._is_reasoning_model() == True
    
    # Test older Claude 3.5 format (should not support reasoning)
    model_35 = AnthropicModel(model="claude-3-5-sonnet-20240620")
    assert model_35._parse_model_version() == (3, 5)
    assert model_35._is_reasoning_model() == False
    
    # Test Claude 4 Opus
    model_opus4 = AnthropicModel(model="claude-opus-4-20250514")
    assert model_opus4._parse_model_version() == (4, 0)
    assert model_opus4._is_reasoning_model() == True
    
    # Test Claude 4 Sonnet
    model_sonnet4 = AnthropicModel(model="claude-sonnet-4-20250514")
    assert model_sonnet4._parse_model_version() == (4, 0)
    assert model_sonnet4._is_reasoning_model() == True
    
    # Test future Claude 4.5 (hypothetical)
    model_45 = AnthropicModel(model="claude-sonnet-4-5-20250614")
    assert model_45._parse_model_version() == (4, 5)
    assert model_45._is_reasoning_model() == True
    
    # Test future Claude 5 (hypothetical)
    model_5 = AnthropicModel(model="claude-opus-5-20250714")
    assert model_5._parse_model_version() == (5, 0)
    assert model_5._is_reasoning_model() == True
    
    # Test future Claude 5.1 (hypothetical)
    model_51 = AnthropicModel(model="claude-sonnet-5-1-20250814")
    assert model_51._parse_model_version() == (5, 1)
    assert model_51._is_reasoning_model() == True
    
    # Test unknown format
    model_unknown = AnthropicModel(model="some-unknown-model")
    assert model_unknown._parse_model_version() == (0, 0)
    assert model_unknown._is_reasoning_model() == False


def test_reasoning_mode_parameter_compatibility():
    """Test that reasoning mode correctly filters incompatible parameters."""
    
    # Test without reasoning mode
    model = AnthropicModel(model="claude-3-7-sonnet-20250219")
    api_params = {
        "model": "claude-3-7-sonnet-20250219",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 1000
    }
    
    # Without thinking parameter, all params should remain
    filtered_params = model._get_compatible_params(api_params)
    assert filtered_params == api_params
    
    # Test with reasoning mode
    model_with_thinking = AnthropicModel(
        model="claude-3-7-sonnet-20250219",
        thinking=ThinkingConfig(type="enabled", budget_tokens=4000)
    )
    
    # With thinking parameter, top_p and top_k should be removed, temperature forced to 1.0
    filtered_params = model_with_thinking._get_compatible_params(api_params)
    expected_params = {
        "model": "claude-3-7-sonnet-20250219",
        "temperature": 1.0,  # temperature is forced to 1.0 for reasoning mode
        "max_tokens": 1000
    }
    assert filtered_params == expected_params
    assert "top_p" not in filtered_params
    assert "top_k" not in filtered_params
    
    # Test that temperature is forced to 1.0 in reasoning mode
    assert filtered_params["temperature"] == 1.0


def test_reasoning_mode_initialization():
    """Test that AnthropicModel can be initialized with thinking parameter."""
    
    thinking_config = ThinkingConfig(type="enabled", budget_tokens=8000)
    
    model = AnthropicModel(
        model="claude-sonnet-4-20250514",
        thinking=thinking_config,
        temperature=0.5,
        max_tokens=2000
    )
    
    assert model.thinking == thinking_config
    assert model.thinking.type == "enabled"
    assert model.thinking.budget_tokens == 8000
    assert model.model == "claude-sonnet-4-20250514"
    assert model.temperature == 0.5
    assert model.max_tokens == 2000
    assert model._is_reasoning_model() == True


def test_reasoning_mode_default_type():
    """Test that ThinkingConfig defaults to type='enabled' when only budget_tokens is set."""
    
    # Test with only budget_tokens specified
    thinking_config = ThinkingConfig(budget_tokens=6000)
    
    model = AnthropicModel(
        model="claude-opus-4-20250514",
        thinking=thinking_config,
        temperature=0.3,
        max_tokens=1500
    )
    
    # Verify defaults
    assert model.thinking.type == "enabled"  # Should default to "enabled"
    assert model.thinking.budget_tokens == 6000
    assert model._is_reasoning_model() == True
    
    # Test that the API parameters are correctly set
    api_params = {
        "model": "claude-opus-4-20250514",
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 30,
        "max_tokens": 1500
    }
    
    filtered_params = model._get_compatible_params(api_params)
    
    # Should remove top_p and top_k for reasoning mode, temperature forced to 1.0
    assert "top_p" not in filtered_params
    assert "top_k" not in filtered_params
    assert filtered_params["temperature"] == 1.0  # Forced to 1.0 in reasoning mode


def test_reasoning_mode_with_claude_sonnet_4():
    """Test reasoning mode functionality with Claude Sonnet 4."""
    from patterpunk.llm.chat import Chat
    
    # Test initialization with thinking parameter
    thinking_config = ThinkingConfig(type="enabled", budget_tokens=4000)
    
    chat = Chat(
        model=AnthropicModel(
            model="claude-sonnet-4-20250514",
            thinking=thinking_config,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=2000
        )
    )
    
    # Verify the model is properly configured
    assert chat.model.thinking == thinking_config
    assert chat.model._is_reasoning_model() == True
    assert chat.model._parse_model_version() == (4, 0)
    
    # Test parameter filtering for API call
    api_params = {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 2000,
        "system": "You are a helpful assistant.",
        "messages": []
    }
    
    # When thinking is enabled, top_p and top_k should be removed
    filtered_params = chat.model._get_compatible_params(api_params)
    
    # Verify reasoning mode constraints are applied
    assert "thinking" not in filtered_params  # thinking added separately in generate_assistant_message
    assert "top_p" not in filtered_params
    assert "top_k" not in filtered_params
    assert "temperature" in filtered_params  # temperature is forced to 1.0
    assert filtered_params["temperature"] == 1.0
    assert filtered_params["max_tokens"] == 2000
    
    # Test with claude-3-7-sonnet as well for comparison
    chat_37 = Chat(
        model=AnthropicModel(
            model="claude-3-7-sonnet-20250219",
            thinking=ThinkingConfig(type="enabled", budget_tokens=2000),
            temperature=0.5
        )
    )
    
    assert chat_37.model._is_reasoning_model() == True
    assert chat_37.model._parse_model_version() == (3, 7)
    
    # Test that non-reasoning models don't apply constraints
    chat_35 = Chat(
        model=AnthropicModel(
            model="claude-3-5-sonnet-20240620",
            temperature=0.5,
            top_p=0.8,
            top_k=50
        )
    )
    
    assert chat_35.model._is_reasoning_model() == False
    assert chat_35.model._parse_model_version() == (3, 5)
    
    # Without thinking parameter, all params should remain
    non_reasoning_params = {
        "temperature": 0.5,
        "top_p": 0.8,
        "top_k": 50,
        "max_tokens": 1000
    }
    filtered_non_reasoning = chat_35.model._get_compatible_params(non_reasoning_params)
    assert filtered_non_reasoning == non_reasoning_params


def test_reasoning_mode_plain_text_response():
    """Test reasoning mode with plain text response from Claude Sonnet 4."""
    from patterpunk.llm.chat import Chat
    
    chat = Chat(
        model=AnthropicModel(
            model="claude-sonnet-4-20250514",
            thinking=ThinkingConfig(budget_tokens=2000),
            max_tokens=4000,  # Must be greater than budget_tokens
            temperature=1.0  # Must be 1.0 when thinking is enabled
        )
    )
    
    response = (
        chat
        .add_message(SystemMessage("You are a helpful math tutor."))
        .add_message(UserMessage("Explain why 2 + 2 = 4 using basic mathematical principles."))
        .complete()
    )
    
    # Verify we got a response
    assert response.latest_message is not None
    assert response.latest_message.content is not None
    assert len(response.latest_message.content.strip()) > 0
    
    # Should mention mathematical concepts
    content_lower = response.latest_message.content.lower()
    assert any(term in content_lower for term in ["addition", "sum", "math", "number", "equal"])


def test_reasoning_mode_structured_output():
    """Test reasoning mode with structured output from Claude 3.7 Sonnet."""
    from patterpunk.llm.chat import Chat
    
    class MathExplanation(BaseModel):
        concept: str = Field(description="The mathematical concept being explained")
        explanation: str = Field(description="Clear explanation of the concept")
        examples: List[str] = Field(description="2-3 simple examples")
        difficulty_level: str = Field(description="beginner, intermediate, or advanced")
        confidence: float = Field(ge=0.0, le=1.0, description="Confidence in explanation accuracy")
    
    chat = Chat(
        model=AnthropicModel(
            model="claude-3-7-sonnet-20250219",
            thinking=ThinkingConfig(budget_tokens=3000),
            max_tokens=5000,  # Must be greater than budget_tokens
            temperature=1.0   # Must be 1.0 when thinking is enabled
        )
    )
    
    response = (
        chat
        .add_message(SystemMessage("You are a math educator. Provide structured explanations."))
        .add_message(UserMessage(
            "Explain the concept of multiplication.",
            structured_output=MathExplanation
        ))
        .complete()
    )
    
    # Verify structured output
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
    """Test reasoning mode with tool calling functionality."""
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
    
    chat = Chat(
        model=AnthropicModel(
            model="claude-opus-4-20250514",
            thinking=ThinkingConfig(budget_tokens=4000),
            max_tokens=6000,  # Must be greater than budget_tokens
            temperature=1.0   # Must be 1.0 when thinking is enabled
        )
    ).with_tools([calculate_area, get_math_fact])
    
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
