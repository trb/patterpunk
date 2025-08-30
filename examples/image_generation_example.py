#!/usr/bin/env python3

import os
from pathlib import Path

from patterpunk.llm.chat import Chat
from patterpunk.llm.models.google import GoogleModel
from patterpunk.llm.messages import UserMessage, SystemMessage
from patterpunk.llm.multimodal import MultimodalChunk
from patterpunk.llm.text import TextChunk
from patterpunk.llm.output_types import OutputType


def generate_single_image():
    print("\n=== Example 1: Generate a Single Image ===\n")
    
    model = GoogleModel(
        model="gemini-2.5-flash-image-preview",
        location="global",
        temperature=0.7
    )
    
    chat = Chat(model=model)
    
    response = (
        chat.add_message(
            SystemMessage("You are a creative digital artist.")
        )
        .add_message(
            UserMessage(
                "Create a beautiful watercolor painting of a Japanese garden "
                "with a red bridge over a koi pond, cherry blossoms, and Mount Fuji in the background"
            )
        )
        .complete(output_types={OutputType.TEXT, OutputType.IMAGE})
    )
    
    print("Text response:", response.latest_message.content[:200] + "...")
    
    if response.latest_message.images:
        image = response.latest_message.images[0]
        output_path = "japanese_garden.png"
        
        with open(output_path, "wb") as f:
            f.write(image.to_bytes())
        
        print(f"Image saved to: {output_path}")
        print(f"Image size: {len(image.to_bytes())} bytes")
        print(f"Media type: {image.media_type}")
    else:
        print("No image was generated")


def edit_existing_image():
    print("\n=== Example 2: Edit an Existing Image ===\n")
    
    # Check if we have a sample image to edit
    sample_image_path = "japanese_garden.png"
    
    if not Path(sample_image_path).exists():
        print(f"Please run generate_single_image() first to create {sample_image_path}")
        return
    
    model = GoogleModel(
        model="gemini-2.5-flash-image-preview",
        location="global",
        temperature=0.7
    )
    
    chat = Chat(model=model)
    
    original_image = MultimodalChunk.from_file(sample_image_path)
    
    response = (
        chat.add_message(
            SystemMessage("You are an expert photo editor.")
        )
        .add_message(
            UserMessage(content=[
                TextChunk(
                    "Transform this image to nighttime with a full moon, "
                    "lanterns lighting the path, and fireflies in the air. "
                    "Keep the same composition but make it magical and atmospheric."
                ),
                original_image
            ])
        )
        .complete(output_types={OutputType.TEXT, OutputType.IMAGE})
    )
    
    print("Edit description:", response.latest_message.content[:200] + "...")
    
    if response.latest_message.images:
        edited_image = response.latest_message.images[0]
        output_path = "japanese_garden_night.png"
        
        with open(output_path, "wb") as f:
            f.write(edited_image.to_bytes())
        
        print(f"Edited image saved to: {output_path}")
        print(f"Image size: {len(edited_image.to_bytes())} bytes")
    else:
        print("No edited image was generated")


def generate_story_with_images():
    print("\n=== Example 3: Generate an Illustrated Story ===\n")
    
    model = GoogleModel(
        model="gemini-2.5-flash-image-preview",
        location="global",
        temperature=0.8
    )
    
    chat = Chat(model=model)
    
    response = (
        chat.add_message(
            SystemMessage(
                "You are a children's book author and illustrator. "
                "Create engaging stories with beautiful illustrations."
            )
        )
        .add_message(
            UserMessage(
                "Create a short 3-part story about a brave little fox exploring a magical forest. "
                "Include an illustration for each part of the story. "
                "Make it whimsical and suitable for young children."
            )
        )
        .complete(output_types={OutputType.TEXT, OutputType.IMAGE})
    )
    
    print("\n--- Story with Illustrations ---\n")
    
    # Print the full text content
    print("Story text:")
    print(response.latest_message.content)
    print()
    
    # Save all images
    for idx, image in enumerate(response.latest_message.images):
        output_path = f"fox_story_part_{idx + 1}.png"
        
        with open(output_path, "wb") as f:
            f.write(image.to_bytes())
        
        print(f"Illustration {idx + 1} saved to: {output_path}")
        print(f"  Size: {len(image.to_bytes())} bytes")
        print(f"  Type: {image.media_type}")
    
    print(f"\nTotal illustrations generated: {len(response.latest_message.images)}")


def verify_generated_image():
    print("\n=== Example 4: Verify Generated Image Content ===\n")
    
    # This example requires OpenAI API key for verification
    if not os.getenv("PP_OPENAI_API_KEY"):
        print("Note: OpenAI API key not found. Skipping image verification example.")
        print("Set PP_OPENAI_API_KEY environment variable to enable this example.")
        return
    
    from patterpunk.llm.models.openai import OpenAiModel
    
    # First, generate an image with specific requirements
    model = GoogleModel(
        model="gemini-2.5-flash-image-preview",
        location="global",
        temperature=0.7
    )
    
    chat = Chat(model=model)
    
    requirements = [
        "A red sports car",
        "Driving on a mountain road",
        "Snow-capped peaks in background",
        "Sunset lighting",
        "Motion blur on wheels"
    ]
    
    prompt = "Create a dynamic photo of: " + ", ".join(requirements)
    
    response = (
        chat.add_message(SystemMessage("You are a professional automotive photographer."))
        .add_message(UserMessage(prompt))
        .complete(output_types={OutputType.TEXT, OutputType.IMAGE})
    )
    
    if not response.latest_message.images:
        print("No image was generated")
        return
    
    generated_image = response.latest_message.images[0]
    
    # Save the generated image
    with open("sports_car.png", "wb") as f:
        f.write(generated_image.to_bytes())
    print("Generated image saved to: sports_car.png")
    
    # Now verify it meets requirements using GPT-4o
    print("\nVerifying image content with GPT-4o...")
    
    verifier_model = OpenAiModel(model="gpt-4o", temperature=0.1)
    verifier_chat = Chat(model=verifier_model)
    
    verification_prompt = f"""
    Check if this image contains ALL of these required elements:
    {chr(10).join(f'- {req}' for req in requirements)}
    
    For each requirement, state YES or NO with a brief explanation.
    End with VERDICT: PASS if all requirements are met, otherwise VERDICT: FAIL.
    """
    
    verification_response = (
        verifier_chat.add_message(
            SystemMessage("You are an expert image analyst. Be precise and thorough.")
        )
        .add_message(
            UserMessage(content=[
                TextChunk(verification_prompt),
                generated_image
            ])
        )
        .complete()
    )
    
    print("\nVerification Results:")
    print(verification_response.latest_message.content)
    
    if "VERDICT: PASS" in verification_response.latest_message.content.upper():
        print("\n✓ Image passed all requirements!")
    else:
        print("\n✗ Image did not meet all requirements")


def main():
    print("=" * 60)
    print("Patterpunk Image Generation Examples")
    print("=" * 60)
    
    # Check for Google credentials
    if not os.getenv("PP_GOOGLE_APPLICATION_CREDENTIALS"):
        print("\n⚠️  Warning: Google credentials not configured!")
        print("Set these environment variables:")
        print("  - PP_GOOGLE_APPLICATION_CREDENTIALS")
        print("  - PP_GEMINI_PROJECT")
        print("  - PP_GEMINI_REGION (set to 'global')")
        print("\nExiting...")
        return
    
    try:
        # Run examples
        generate_single_image()
        edit_existing_image()
        generate_story_with_images()
        verify_generated_image()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check the generated image files in the current directory.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Valid Google Cloud credentials")
        print("2. Access to gemini-2.5-flash-image-preview model")
        print("3. Correct project ID and region settings")


if __name__ == "__main__":
    main()