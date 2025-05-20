import logging
from typing import List, Optional
from pydantic import BaseModel

import patterpunk.lib.extract_json
from patterpunk.llm.chat import Chat
from patterpunk.llm.models.ollama import OllamaModel
from patterpunk.llm.messages import SystemMessage, UserMessage
from patterpunk.logger import logger, logger_llm


def run_basic_test(model: str):
    chat = Chat(model=OllamaModel(model=model, temperature=0.01))

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
    print(patterpunk.lib.extract_json.extract_json())


def test_basic():
    logger.setLevel(logging.ERROR)
    logger_llm.setLevel(logging.ERROR)

    for model in OllamaModel.get_available_models():
        print()
        print(
            "=============================================================================="
        )
        print(
            "=============================================================================="
        )
        print(f"==================={model}=======================")
        print(
            "=============================================================================="
        )
        print(
            "=============================================================================="
        )
        print()
        for x in range(10):
            print()
            print("---------------------")
            print(f"--------run {x}--------")
            print("---------------------")
            print()
            run_basic_test(model)
        print()
        print()
        print()


def test_structured_output():
    # Define Pydantic models for structured output
    class Person(BaseModel):
        name: str
        role: str
        description: Optional[str] = None

    class Topic(BaseModel):
        name: str
        importance: int  # 1-10 scale
        related_keywords: List[str]

    class ArticleSection(BaseModel):
        title: str
        content_summary: str
        sentiment: Optional[str] = None
        key_points: List[str]

    class NewsAnalysis(BaseModel):
        article_title: str
        publication_date: Optional[str] = None
        source: str
        author: Optional[Person] = None
        key_people: List[Person]
        main_topics: List[Topic]
        sections: List[ArticleSection]
        overall_sentiment: str
        factual_accuracy_score: Optional[int] = None

    # Test article text
    article_text = """
    TECH BREAKTHROUGH: QUANTUM COMPUTING REACHES NEW MILESTONE
    
    By Sarah Chen | March 15, 2024 | TechFuture Magazine
    
    In a significant breakthrough announced yesterday, researchers at QuantumWave Labs have 
    successfully demonstrated quantum supremacy in a practical application for the first time. 
    The team, led by Dr. James Rodriguez, achieved stable quantum operations at room temperature, 
    a feat previously thought impossible before 2030.
    
    "This represents a fundamental shift in quantum computing viability," said Rodriguez during 
    the press conference. "We've essentially compressed the timeline for practical quantum 
    applications by several years."
    
    Industry Impact
    
    The breakthrough has sent shockwaves through the tech industry. Major players including 
    Google, IBM, and Microsoft have already announced increased funding for their quantum 
    divisions. Tech stocks surged following the announcement, with QuantumWave's parent 
    company seeing a 27% increase in share value.
    
    Dr. Lisa Patel, quantum computing expert at MIT who wasn't involved in the research, 
    called the development "genuinely revolutionary" but cautioned that "significant 
    challenges remain before consumer applications become viable."
    
    Security Implications
    
    Cybersecurity experts have expressed concerns about the accelerated timeline. "Current 
    encryption standards were developed with the assumption that quantum computing at this 
    level was at least a decade away," noted cybersecurity analyst Marcus Johnson.
    
    The National Security Agency released a brief statement acknowledging the development 
    and reiterating its commitment to quantum-resistant encryption standards.
    
    What's Next
    
    QuantumWave Labs has announced plans to publish their complete methodology in next 
    month's issue of Nature Quantum. They've also secured $150 million in additional 
    funding to scale their prototype.
    
    For everyday consumers, the impact won't be immediate, but experts suggest quantum-enhanced 
    products could reach the market within 3-5 years, significantly earlier than previous 
    estimates of 7-10 years.
    """

    # Test with both specified models
    models_to_test = [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "meta.llama3-70b-instruct-v1:0",
    ]

    for model_name in models_to_test:
        print(f"\nTesting structured output with model: {model_name}")

        # Create chat with model and structured output
        chat = Chat(model=OllamaModel(model=model_name, temperature=0.1))

        # Add system message with instructions
        chat = chat.add_message(
            SystemMessage(
                """
                You are an expert news analyst. Your task is to analyze news articles and extract 
                key information in a structured format. Analyze the article thoroughly and identify:
                
                - The article title and publication details
                - Key people mentioned and their roles
                - Main topics discussed and their importance
                - Different sections of the article with summaries
                - Overall sentiment and factual accuracy
                
                Be thorough in your analysis but stick to information that's actually in the article.
                If information is not provided in the article, indicate that it's missing.
                """
            )
        )

        # Add user message with the article
        chat = chat.add_message(
            UserMessage(
                f"Please analyze this news article:\n\n{article_text}",
                structured_output=NewsAnalysis,
            )
        )

        # Complete the chat
        chat = chat.complete()

        # Get the parsed output
        analysis = chat.parsed_output

        # Print the result
        print(f"Successfully parsed structured output: {analysis}")

        # Assertions to verify the output
        assert (
            analysis.article_title
            == "TECH BREAKTHROUGH: QUANTUM COMPUTING REACHES NEW MILESTONE"
        )
        assert analysis.source == "TechFuture Magazine"
        assert analysis.publication_date == "March 15, 2024"

        # Verify key people extraction
        assert len(analysis.key_people) >= 3  # At least Rodriguez, Patel, and Johnson

        # Find Dr. Rodriguez in key people
        rodriguez = next(
            (p for p in analysis.key_people if "Rodriguez" in p.name), None
        )
        assert rodriguez is not None
        assert (
            "lead" in rodriguez.role.lower() or "researcher" in rodriguez.role.lower()
        )

        # Verify topics
        assert len(analysis.main_topics) >= 2  # At least quantum computing and security

        # Verify sections
        assert len(analysis.sections) >= 3  # Should have multiple sections

        # Verify sentiment is present
        assert analysis.overall_sentiment is not None

        # Verify optional fields
        # The author should be identified as Sarah Chen
        assert analysis.author is not None
        assert analysis.author.name == "Sarah Chen"

        print(f"All assertions passed for model: {model_name}")


"""
@todo Add a test case that tests the structured output support with the models anthropic.claude-3-sonnet-20240229-v1:0 and meta.llama3-70b-instruct-v1:0
Use a relatively complex structured response format so we can ensure that the system can handle nested data structures.
Use pydantic to create the models.

Keep everything contained in the test function. Don't make the model too long, we're looking for nested data structures
and objects and lists, etc - complex, not long. And don't go overboard with the complexity either, just..you know, not
a simple json object with three keys.

Test test should use the Chat parsed_output to assert that the returned object extracted information correctly.

To do so, you'll have to generate a source text that covers the fields for the model. Don't give the model any
specific json format instructions - we want to test that the library handles that. However, _do_ make sure that we test
fields that should be None actually end up as None. I.e. if the model requests a field that the source data does not
provide any information for, it should end up as None - makes sense, right?

Just make sure we have a good test that provides a model, a source text, and ensure that the library correctly extracts
information from the source text into a python object. Don't go overboard on the complexity, but also don't baby the
model.

Do create good prompts for the model, just don't tell it exactly what schema to generate.
"""
