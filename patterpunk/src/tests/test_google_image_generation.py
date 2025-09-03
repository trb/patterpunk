import pytest
from typing import List, Union
import base64
import io

from patterpunk.config.providers.google import GOOGLE_APPLICATION_CREDENTIALS
from patterpunk.config.providers.openai import OPENAI_API_KEY
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage, AssistantMessage, SystemMessage
from patterpunk.llm.multimodal import MultimodalChunk
from patterpunk.llm.text import TextChunk
from patterpunk.llm.output_types import OutputType
from tests.test_utils import get_resource


def verify_image_content(
    image_chunk: MultimodalChunk,
    expected_features: List[str],
    should_not_contain: List[str] = None,
) -> tuple[bool, str]:
    if not OPENAI_API_KEY:
        pytest.skip("OpenAI credentials not available for image verification")

    verifier_model = OpenAiModel(model="gpt-4o", temperature=0.1)

    verifier_chat = Chat(model=verifier_model)

    verification_prompt = f"""Analyze this image and check for these features:

Expected to be present:
{chr(10).join(f'- {feature}' for feature in expected_features)}

{"Expected NOT to be present:" + chr(10) + chr(10).join(f'- {feature}' for feature in should_not_contain) if should_not_contain else ""}

For each feature, briefly explain if it's present or not.
End with: VERDICT: PASS (if all expected features are found and no forbidden features are present) or VERDICT: FAIL"""

    response = (
        verifier_chat.add_message(SystemMessage("You are an expert image analyst."))
        .add_message(UserMessage(content=[TextChunk(verification_prompt), image_chunk]))
        .complete()
    )

    analysis = response.latest_message.content
    success = "PASS" in analysis.upper()

    return success, analysis


@pytest.mark.integration
def test_google_generate_new_image():
    if not GOOGLE_APPLICATION_CREDENTIALS:
        pytest.skip("Google credentials not available")

    model = GoogleModel(
        model="gemini-2.5-flash-image-preview", location="global", temperature=0.7
    )

    chat = Chat(model=model)

    response = (
        chat.add_message(
            SystemMessage(
                "You are an expert digital artist who creates vivid, detailed images based on descriptions."
            )
        )
        .add_message(
            UserMessage(
                content=[
                    TextChunk(
                        """Create a detailed, photorealistic image of a futuristic floating city at sunset. 
                    The city should have:
                    - Sleek glass towers with curved architecture
                    - Flying vehicles moving between buildings
                    - Warm orange and purple sunset colors reflecting off the glass
                    - Some greenery integrated into the buildings (vertical gardens)
                    - Clouds below the city showing it's floating high in the sky
                    
                    Make it look cinematic and breathtaking."""
                    )
                ]
            )
        )
        .complete(output_types={OutputType.TEXT, OutputType.IMAGE})
    )

    message = response.latest_message
    assert isinstance(message, AssistantMessage)

    assert hasattr(
        message, "_raw_content"
    ), "AssistantMessage should have _raw_content attribute"
    assert isinstance(
        message._raw_content, list
    ), "Raw content should be a list for multimodal response"

    text_chunks = message.texts
    image_chunks = message.images

    assert (
        len(text_chunks) > 0
    ), "Response should contain descriptive text about the generated image"
    assert len(image_chunks) > 0, "Response should contain at least one generated image"

    for text_chunk in text_chunks:
        assert isinstance(text_chunk, (str, TextChunk))
        if isinstance(text_chunk, TextChunk):
            text_content = text_chunk.content
        else:
            text_content = text_chunk
        assert len(text_content) > 10, "Text description should be meaningful"

    for image_chunk in image_chunks:
        assert isinstance(
            image_chunk, MultimodalChunk
        ), f"Image chunk should be MultimodalChunk, got {type(image_chunk)}"
        assert image_chunk.media_type is not None, "Image should have a media type"
        assert image_chunk.media_type.startswith(
            "image/"
        ), f"Media type should be image/*, got {image_chunk.media_type}"

        image_bytes = image_chunk.to_bytes()
        assert (
            len(image_bytes) > 1000
        ), "Image should have substantial data (at least 1KB)"

        image_header = image_bytes[:10]
        valid_headers = [
            b"\xff\xd8\xff",
            b"\x89PNG\r\n\x1a\n",
            b"GIF87a",
            b"GIF89a",
            b"RIFF",
            b"<svg",
            b"II*\x00",
            b"MM\x00*",
        ]

        is_valid_image = any(
            image_header.startswith(header) for header in valid_headers
        )
        assert (
            is_valid_image
        ), f"Image should have valid image file header, got {image_header[:4]}"

        print(
            f"Generated futuristic city image: {len(image_bytes)} bytes, header={image_header[:4]}"
        )

        with open("/tmp/generated_futuristic_city.png", "wb") as f:
            f.write(image_bytes)
        print("Saved generated image to /tmp/generated_futuristic_city.png")

        print("\nVerifying image content with GPT-4o...")
        verification_success, verification_analysis = verify_image_content(
            image_chunk,
            expected_features=[
                "Futuristic city with modern architecture",
                "Tall buildings or towers",
                "City floating or elevated in the sky",
                "Clouds below or around the city",
                "Sunset colors (orange/purple/warm tones)",
                "Flying vehicles or futuristic transportation",
            ],
            should_not_contain=[
                "Medieval or ancient architecture",
                "Underwater scene",
                "Rural countryside",
            ],
        )

        print(f"\nImage verification analysis:\n{verification_analysis}")
        assert (
            verification_success
        ), f"Generated image does not match expected futuristic floating city. Analysis: {verification_analysis}"

    combined_text = message.content
    assert (
        "city" in combined_text.lower()
        or "futuristic" in combined_text.lower()
        or "floating" in combined_text.lower()
    ), "Response should mention elements from the prompt"


@pytest.mark.integration
def test_google_edit_existing_image_ducks_to_pelicans():
    if not GOOGLE_APPLICATION_CREDENTIALS:
        pytest.skip("Google credentials not available")

    model = GoogleModel(
        model="gemini-2.5-flash-image-preview", location="global", temperature=0.7
    )

    chat = Chat(model=model)

    ducks_image = MultimodalChunk.from_file(get_resource("ducks_pond.jpg"))

    response = (
        chat.add_message(
            SystemMessage(
                """You are an expert photo editor who can seamlessly modify images while 
            maintaining photorealism and preserving the original scene composition."""
            )
        )
        .add_message(
            UserMessage(
                content=[
                    TextChunk(
                        """Look at this image carefully. I want you to:
                    1. Replace all the ducks in this pond image with pelicans
                    2. Keep the same pond, water, and background environment
                    3. Maintain the same lighting and atmosphere
                    4. Make the pelicans look natural and photorealistic
                    5. Preserve the overall composition and feel of the original image
                    
                    Generate a new version of this image with pelicans instead of ducks.
                    Also provide a description of what changes you made."""
                    ),
                    ducks_image,
                ]
            )
        )
        .complete(output_types={OutputType.TEXT, OutputType.IMAGE})
    )

    message = response.latest_message
    assert isinstance(message, AssistantMessage)

    assert hasattr(
        message, "_raw_content"
    ), "AssistantMessage should have _raw_content attribute"
    assert isinstance(
        message._raw_content, list
    ), "Raw content should be a list for multimodal response"

    text_chunks = message.texts
    image_chunks = message.images

    assert (
        len(text_chunks) > 0
    ), "Response should contain text describing the edits made"
    assert (
        len(image_chunks) > 0
    ), "Response should contain the edited image with pelicans"

    for text_chunk in text_chunks:
        if isinstance(text_chunk, TextChunk):
            text_content = text_chunk.content
        else:
            text_content = text_chunk
        assert (
            len(text_content) > 10
        ), "Text description should describe the editing process"

    combined_text = message.content.lower()
    assert (
        "pelican" in combined_text
        or "replaced" in combined_text
        or "edit" in combined_text
        or "change" in combined_text
    ), "Response text should mention pelicans or the editing/replacement process"

    for idx, image_chunk in enumerate(image_chunks):
        assert isinstance(
            image_chunk, MultimodalChunk
        ), f"Image chunk should be MultimodalChunk, got {type(image_chunk)}"
        assert (
            image_chunk.media_type is not None
        ), "Edited image should have a media type"
        assert image_chunk.media_type.startswith(
            "image/"
        ), f"Media type should be image/*, got {image_chunk.media_type}"

        image_bytes = image_chunk.to_bytes()
        assert (
            len(image_bytes) > 1000
        ), "Edited image should have substantial data (at least 1KB)"

        image_header = image_bytes[:10]
        valid_headers = [
            b"\xff\xd8\xff",
            b"\x89PNG\r\n\x1a\n",
            b"GIF87a",
            b"GIF89a",
            b"RIFF",
            b"<svg",
            b"II*\x00",
            b"MM\x00*",
        ]

        is_valid_image = any(
            image_header.startswith(header) for header in valid_headers
        )
        assert (
            is_valid_image
        ), f"Edited image should have valid image file header, got {image_header[:4]}"

        print(
            f"Edited pelican image {idx}: {len(image_bytes)} bytes, header={image_header[:4]}"
        )

        with open(f"/tmp/edited_pelicans_pond_{idx}.png", "wb") as f:
            f.write(image_bytes)
        print(f"Saved edited image to /tmp/edited_pelicans_pond_{idx}.png")

        print(f"\nVerifying edited image {idx} content with GPT-4o...")
        verification_success, verification_analysis = verify_image_content(
            image_chunk,
            expected_features=[
                "Pelicans (large birds with long beaks)",
                "Pond or water body",
                "Birds near or on the water",
            ],
            should_not_contain=["Ducks", "Geese", "Swans"],
        )

        print(f"\nEdited image {idx} verification analysis:\n{verification_analysis}")
        assert (
            verification_success
        ), f"Edited image does not show pelicans instead of ducks. Analysis: {verification_analysis}"


@pytest.mark.integration
def test_google_multimodal_generation_with_multiple_images():
    if not GOOGLE_APPLICATION_CREDENTIALS:
        pytest.skip("Google credentials not available")

    model = GoogleModel(
        model="gemini-2.5-flash-image-preview", location="global", temperature=0.7
    )

    chat = Chat(model=model)

    response = (
        chat.add_message(
            SystemMessage(
                "You are a creative visual storyteller who creates illustrated stories with multiple images."
            )
        )
        .add_message(
            UserMessage(
                content=[
                    TextChunk(
                        """Create a short illustrated story about a day in the life of a robot chef. 
                    Include:
                    1. An image of the robot chef waking up in the morning
                    2. An image of the robot preparing ingredients 
                    3. An image of the robot cooking a spectacular dish
                    4. An image of happy customers enjoying the meal
                    
                    Provide narrative text between each image to tell the story.
                    Make the images colorful, whimsical, and consistent in style."""
                    )
                ]
            )
        )
        .complete(output_types={OutputType.TEXT, OutputType.IMAGE})
    )

    message = response.latest_message
    assert isinstance(message, AssistantMessage)

    text_chunks = message.texts
    image_chunks = message.images

    assert len(text_chunks) > 0, "Response should contain story narrative text"
    assert (
        len(image_chunks) >= 2
    ), "Response should contain multiple images for the story"

    print(
        f"Generated story with {len(text_chunks)} text segments and {len(image_chunks)} images"
    )

    for idx, image_chunk in enumerate(image_chunks):
        assert isinstance(image_chunk, MultimodalChunk)
        image_bytes = image_chunk.to_bytes()
        assert (
            len(image_bytes) > 1000
        ), f"Story image {idx} should have substantial data"

        image_header = image_bytes[:10]
        valid_headers = [
            b"\xff\xd8\xff",
            b"\x89PNG\r\n\x1a\n",
            b"GIF87a",
            b"GIF89a",
            b"RIFF",
            b"<svg",
            b"II*\x00",
            b"MM\x00*",
        ]

        is_valid_image = any(
            image_header.startswith(header) for header in valid_headers
        )
        assert (
            is_valid_image
        ), f"Story image {idx} should have valid image file header, got {image_header[:4]}"

        print(f"Story image {idx}: {len(image_bytes)} bytes")
        with open(f"/tmp/robot_chef_story_{idx}.png", "wb") as f:
            f.write(image_bytes)
        print(f"Saved story image {idx} to /tmp/robot_chef_story_{idx}.png")

        print(f"\nVerifying story image {idx} content with GPT-4o...")
        verification_success, verification_analysis = verify_image_content(
            image_chunk,
            expected_features=[
                "Robot or mechanical character",
                "Cooking or kitchen scene",
                "Colorful whimsical illustration",
            ],
            should_not_contain=["Photographic realism", "Dark or horror theme"],
        )

        print(f"\nStory image {idx} verification analysis:\n{verification_analysis}")
        assert (
            verification_success
        ), f"Story image {idx} does not match robot chef theme. Analysis: {verification_analysis}"

    combined_text = message.content.lower()
    assert (
        "robot" in combined_text
        or "chef" in combined_text
        or "cooking" in combined_text
    ), "Story text should relate to the robot chef theme"


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
