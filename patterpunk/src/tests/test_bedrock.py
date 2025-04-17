
import pytest
from pydantic import BaseModel, Field
from typing import List, Optional

from patterpunk.llm.models.bedrock import BedrockModel
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage, SystemMessage


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "meta.llama3-8b-instruct-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
    ],
)
def test_simple_bedrock(model_id):
    bedrock = BedrockModel(model_id=model_id, temperature=0.1, top_p=0.98)

    print()
    print("Bedrock Models")
    print(BedrockModel.get_available_models())
    print()

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
        reviewer_name: str = Field(description="Name of the person who wrote the review")
        pros: List[str] = Field(description="List of positive aspects of the product")
        cons: List[str] = Field(description="List of negative aspects of the product")
        key_features: List[ProductFeature] = Field(description="List of key features of the product")
        warranty_period: Optional[str] = Field(description="Length of warranty if mentioned")
        competitor_comparison: Optional[str] = Field(description="Comparison to competitor products if mentioned")
        recommended: bool = Field(description="Whether the reviewer recommends the product")

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
    assert any(feature.name == "noise cancellation" for feature in parsed_output.key_features) or \
           any("noise" in feature.name.lower() for feature in parsed_output.key_features)
    assert parsed_output.warranty_period is None  # Not mentioned in the review
    assert parsed_output.competitor_comparison is None  # Not mentioned in the review
    assert parsed_output.recommended is True
