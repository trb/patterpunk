"""
Azure OpenAI Example

This example demonstrates how to use Azure OpenAI with patterpunk.

Prerequisites:
1. Azure OpenAI resource created in Azure Portal
2. Model deployed with a deployment name (e.g., "gpt-4o")
3. Environment variables set:
   - PP_AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   - PP_AZURE_OPENAI_API_KEY=your-api-key
   - PP_AZURE_OPENAI_API_VERSION=2024-10-21 (or later)

Note: Azure OpenAI uses deployment names instead of model names.
The deployment name is what you set when deploying a model in Azure Portal,
not the underlying model name (like "gpt-4o").
"""

from patterpunk.llm.models.azure_openai import AzureOpenAiModel
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import SystemMessage, UserMessage
from patterpunk.llm.thinking import ThinkingConfig
from pydantic import BaseModel, Field


def basic_chat_example():
    """Basic chat completion with Azure OpenAI"""
    print("\n=== Basic Chat Example ===")

    # Replace "gpt-4o" with your actual Azure deployment name
    chat = Chat(model=AzureOpenAiModel(deployment_name="gpt-4o", temperature=0.7))

    response = (
        chat.add_message(SystemMessage("You are a helpful assistant."))
        .add_message(UserMessage("What are the three primary colors?"))
        .complete()
    )

    print(f"Response: {response.latest_message.content}")


def structured_output_example():
    """Structured output with Azure OpenAI"""
    print("\n=== Structured Output Example ===")

    class ColorInfo(BaseModel):
        primary_colors: list[str] = Field(description="List of primary colors")
        description: str = Field(description="Brief explanation")

    chat = Chat(model=AzureOpenAiModel(deployment_name="gpt-4o", temperature=0.0))

    response = (
        chat.add_message(SystemMessage("You are a helpful assistant."))
        .add_message(
            UserMessage("What are the three primary colors?", structured_output=ColorInfo)
        )
        .complete()
    )

    parsed = response.parsed_output
    print(f"Primary Colors: {parsed.primary_colors}")
    print(f"Description: {parsed.description}")


def reasoning_model_example():
    """Using reasoning models (o-series) with Azure OpenAI"""
    print("\n=== Reasoning Model Example ===")

    # Note: Requires an o-series model deployment (e.g., o3-mini)
    chat = Chat(
        model=AzureOpenAiModel(
            deployment_name="o3-mini",  # Your o3-mini deployment name
            thinking_config=ThinkingConfig(effort="high"),
        )
    )

    response = (
        chat.add_message(
            SystemMessage("You are a logical reasoning assistant.")
        )
        .add_message(
            UserMessage(
                "If all roses are flowers and some flowers fade quickly, "
                "what can we conclude about roses?"
            )
        )
        .complete()
    )

    print(f"Response: {response.latest_message.content}")


def tool_calling_example():
    """Tool calling with Azure OpenAI"""
    print("\n=== Tool Calling Example ===")

    def get_weather(location: str) -> str:
        """Get the current weather for a location.

        Args:
            location: The city and state, e.g., 'San Francisco, CA'

        Returns:
            str: Weather description
        """
        # Mock implementation
        return f"The weather in {location} is sunny and 72°F"

    chat = Chat(model=AzureOpenAiModel(deployment_name="gpt-4o", temperature=0.0))

    response = (
        chat.add_message(SystemMessage("You are a helpful weather assistant."))
        .with_tools([get_weather])
        .add_message(UserMessage("What's the weather in Seattle?"))
        .complete()
        .execute_tool_calls()
        .complete()
    )

    print(f"Response: {response.latest_message.content}")


def multimodal_example():
    """Multimodal input with Azure OpenAI"""
    print("\n=== Multimodal Example ===")

    from patterpunk.llm.multimodal import MultimodalChunk

    # Requires a vision-capable deployment (e.g., gpt-4o)
    chat = Chat(model=AzureOpenAiModel(deployment_name="gpt-4o", temperature=0.0))

    # Example with URL
    response = (
        chat.add_message(SystemMessage("Describe the image concisely."))
        .add_message(
            UserMessage(
                content=[
                    "What do you see in this image?",
                    MultimodalChunk.from_url(
                        "https://example.com/image.jpg", media_type="image/jpeg"
                    ),
                ]
            )
        )
        .complete()
    )

    print(f"Response: {response.latest_message.content}")

    # Example with local file
    # response = (
    #     chat.add_message(SystemMessage("Describe the image concisely."))
    #     .add_message(
    #         UserMessage(
    #             content=[
    #                 "What do you see in this image?",
    #                 MultimodalChunk.from_file("/path/to/local/image.jpg"),
    #             ]
    #         )
    #     )
    #     .complete()
    # )


if __name__ == "__main__":
    # Run examples
    # Comment out examples you don't want to run

    basic_chat_example()
    structured_output_example()

    # Uncomment if you have o-series deployment
    # reasoning_model_example()

    # Uncomment to test tool calling
    # tool_calling_example()

    # Uncomment to test multimodal
    # multimodal_example()
