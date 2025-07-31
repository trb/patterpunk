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
from patterpunk.llm.multimodal import MultimodalChunk
from patterpunk.llm.text import TextChunk

response = Chat().add_message(
    UserMessage([
        TextChunk("Analyze this image:"),
        MultimodalChunk.from_file("screenshot.png")
    ])
).complete()
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
- **AWS Bedrock** (`PP_AWS_*`) - Multiple model families (Claude, Llama, Mistral, Titan)
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

- **[Tool Calling](TOOL_CALLING.md)** - Function conversion, MCP servers, execution flows
- **[Promtp Caching](PROMPT_CACHING.md)** - Images, files, content chunking, provider support
- **[Multimodal Content](MULTIMODAL.md)** - Images, files, content chunking, provider support
- **[Agent Workflows](AGENTS.md)** - Sequential chains, parallel execution, type-safe agents

## Design Philosophy

Patterpunk prioritizes **developer experience** through:

1. **Provider Agnostic** - Write once, run on any LLM provider
2. **Type Safe** - Full TypeScript-like typing with Pydantic integration
3. **Immutable** - Functional composition without side effects
4. **Modular** - Use only the features you need
5. **Production Ready** - Error handling, retries, cost optimization

## Contributing

Patterpunk is open source. Contributions welcome for new providers, features, and improvements.
