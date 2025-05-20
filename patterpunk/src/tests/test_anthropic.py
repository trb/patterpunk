from pydantic import BaseModel, Field
from typing import List, Optional
from patterpunk.llm.chat import Chat
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.messages import SystemMessage, UserMessage


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
        topics: List[str] = Field(min_items=3, max_items=10)
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
