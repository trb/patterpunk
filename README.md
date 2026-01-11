# Patterpunk

**Comprehensive LLM provider abstraction** with unified interface across OpenAI, Anthropic, AWS Bedrock, Google Vertex AI, and Ollama. Built for production with chainable immutable API, native tool calling, multimodal content, and Model Context Protocol (MCP) integration.

## Quick Start

```bash
pip install patterpunk
```

```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage

# Simple conversation
response = Chat().add_message(
    UserMessage("What is the capital of France?")
).complete()

print(response.latest_message.content)
```

### Streaming

Stream responses token-by-token for real-time display:

```python
import asyncio
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage

async def main():
    chat = Chat().add_message(UserMessage("Write a haiku about coding"))

    async with chat.complete_stream() as stream:
        async for content in stream.content:
            print(content, end="\r")  # Each yield is accumulated text

    final_chat = await stream.chat
    print(f"\nComplete: {final_chat.latest_message.content}")

asyncio.run(main())
```

The streaming API works uniformly across all providers. For advanced patterns including thinking/reasoning streams, delta iteration, and tool calling with streaming, see [STREAMING.md](STREAMING.md).

### Tool Calling

```python
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72°F in {city}"

# Automatic function-to-tool conversion
chat = Chat().with_tools([get_weather])
response = chat.add_message(
    UserMessage("What's the weather in Paris?")
).complete()

# Handle tool calls manually (MCP tools execute automatically)
if response.is_latest_message_tool_call:
    # Tool call message contains function calls to execute
    print(response.latest_message.tool_calls)
```

### Multimodal Content

```python
from patterpunk.llm.chunks import MultimodalChunk, TextChunk

response = Chat().add_message(
    UserMessage([
        TextChunk("Analyze this image:"),
        MultimodalChunk.from_file("screenshot.png")
    ])
).complete()
```

### Image Generation

```python
from patterpunk.llm.output_types import OutputType
from patterpunk.llm.models.google import GoogleModel

# Generate single image
model = GoogleModel(model="gemini-2.5-flash-image-preview", location="global")
response = Chat(model).add_message(
    UserMessage("Create a futuristic city at sunset")
).complete(output_types={OutputType.TEXT, OutputType.IMAGE})

image_bytes = response.latest_message.images[0].to_bytes()

# Generate multiple images with text
response = Chat(model).add_message(
    UserMessage("Create a 3-panel comic about a robot learning to cook")
).complete(output_types={OutputType.TEXT, OutputType.IMAGE})

for i, img in enumerate(response.latest_message.images):
    with open(f"panel_{i}.png", "wb") as f:
        f.write(img.to_bytes())

# Edit existing image
response = Chat(model).add_message(
    UserMessage([
        TextChunk("Replace the cars with flying vehicles"),
        MultimodalChunk.from_file("city.jpg")
    ])
).complete(output_types={OutputType.TEXT, OutputType.IMAGE})
```

### Structured Output & Reasoning

```python
from pydantic import BaseModel
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.llm.models.anthropic import AnthropicModel

class WeatherReport(BaseModel):
    location: str
    temperature: float
    conditions: str

# Structured output with reasoning mode
model = AnthropicModel(
    model="claude-sonnet-4-20250514",
    thinking=ThinkingConfig(effort="high")
)

response = Chat(model=model).add_message(
    UserMessage(
        "Analyze the weather in Tokyo",
        structured_output=WeatherReport
    )
).complete()

weather_data = response.parsed_output  # Typed WeatherReport instance
```

## Architecture

Patterpunk uses a **chainable immutable interface** where every operation returns new instances, enabling conversation branching and functional composition. Provider-specific code is isolated in model implementations, ensuring consistent behavior across all LLM providers.

```python
# Immutable chaining enables branching
base_chat = Chat().add_message(SystemMessage("You are helpful")).complete()
branch_a = base_chat.add_message(UserMessage("Tell me about AI"))
branch_b = base_chat.add_message(UserMessage("Tell me about ML"))
```

## Supported Providers

All providers implement the same unified interface with automatic credential detection:

- **OpenAI** (`PP_OPENAI_API_KEY`) - GPT models with native tool calling, reasoning support
- **Anthropic** (`PP_ANTHROPIC_API_KEY`) - Claude models with tool use, thinking modes
- **AWS Bedrock** (`PP_AWS_*`) - Multiple model families (Claude, Llama, Mistral, Titan) - [Setup Guide](#aws-bedrock-setup)
- **Google Vertex AI** (`PP_GOOGLE_*`) - Gemini models with tool calling
- **Ollama** (`PP_OLLAMA_API_ENDPOINT`) - Local model serving

```python
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.models.anthropic import AnthropicModel

# Same interface across all providers
openai_model = OpenAiModel(model="gpt-4.1")
anthropic_model = AnthropicModel(model="claude-sonnet-4-20250514")
```

## Comprehensive Documentation

For detailed information on specific features:

- **[Streaming](STREAMING.md)** - Async streaming, thinking/reasoning, tool calling with streams
- **[Tool Calling](TOOL_CALLING.md)** - Function conversion, MCP servers, execution flows
- **[Prompt Caching](PROMPT_CACHING.md)** - Cache control, provider support, cost optimization
- **[Multimodal Content](MULTIMODAL.md)** - Images, files, content chunking, provider support
- **[Agent Workflows](AGENTS.md)** - Sequential chains, parallel execution, type-safe agents

## AWS Bedrock Setup

AWS Bedrock requires additional setup for Claude models beyond standard IAM permissions.

### Environment Variables

```bash
export PP_AWS_ACCESS_KEY_ID="your-access-key"
export PP_AWS_SECRET_ACCESS_KEY="your-secret-key"
export PP_AWS_REGION="us-east-1"  # or your preferred region
```

### AWS Marketplace Subscription

Claude models on Bedrock are served through AWS Marketplace and require a one-time subscription activation per model family. Even with proper IAM permissions, you'll get `AccessDeniedException` until this is done.

**Required IAM permissions for subscription:**

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "aws-marketplace:ViewSubscriptions",
                "aws-marketplace:Subscribe"
            ],
            "Resource": "*"
        }
    ]
}
```

**Activation process:**

1. Add the above IAM permissions to your user/role
2. Invoke each model once (via API or AWS Console Bedrock Playground)
3. Wait for AWS Marketplace confirmation email (typically immediate)
4. The subscription is now active account-wide for all users

**Available Claude 4.5 inference profile IDs:**

| Model | Inference Profile ID |
|-------|---------------------|
| Claude Sonnet 4.5 | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| Claude Opus 4.5 | `us.anthropic.claude-opus-4-5-20251101-v1:0` |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |

```python
from patterpunk.llm.models.bedrock import BedrockModel

model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
```

## Design Philosophy

Patterpunk prioritizes **developer experience** through:

1. **Provider Agnostic** - Write once, run on any LLM provider
2. **Type Safe** - Full TypeScript-like typing with Pydantic integration
3. **Immutable** - Functional composition without side effects
4. **Modular** - Use only the features you need
5. **Production Ready** - Error handling, retries, cost optimization
