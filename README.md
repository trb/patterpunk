# Patterpunk

LLM provider abstraction library with unified interface across OpenAI, Anthropic, AWS Bedrock, Google Vertex AI, and Ollama. Features chainable immutable API, native tool calling, and Model Context Protocol (MCP) integration.

## Installation

```bash
pip install patterpunk
```

Optional dependencies for MCP integration:
```bash
pip install patterpunk[mcp]  # For MCP server support
```

## Quick Start

### Intro

```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage
from patterpunk.llm.models.openai import OpenAiModel

# Create model and chat
model = OpenAiModel(model="gpt-4.1")
chat = Chat(model=model)

# Have a conversation  
response = chat.add_message(
    UserMessage("What is the capital of France?")
).complete()

print(response.latest_message.content)
```

### Reasoning Mode

Enable extended thinking for Anthropic models:

```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage
from patterpunk.llm.models.anthropic import AnthropicModel, ThinkingConfig

# Claude 3.7 Sonnet with extended thinking
model = AnthropicModel(
    model="claude-3-7-sonnet-20250219",
    thinking=ThinkingConfig(budget_tokens=5000)
)

response = Chat(model=model).add_message(
    UserMessage("Solve this step by step: If a train travels 120 miles in 2 hours, then speeds up by 25%, how long to travel 200 miles?")
).complete()

print(response.latest_message.content)
```

### Tool Calling and MCP Servers

```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny, 72°F in {city}"

# Function tools
chat = Chat().with_tools([get_weather])
response = chat.add_message(
    UserMessage("What's the weather in Paris?")
).complete()

# MCP servers  
from patterpunk.lib.mcp import MCPServerConfig

weather_server = MCPServerConfig(
    name="weather",
    url="http://localhost:8000/mcp"
)

chat = Chat().with_mcp_servers([weather_server])
response = chat.add_message(
    UserMessage("Get weather data")
).complete()
```

## Core Usage

### Chat Class

The `Chat` class provides a chainable immutable interface for conversations:

```python
from patterpunk.llm.chat import Chat

# Create chat instance
chat = Chat()

# Add messages and get responses (returns new Chat instance)
response = chat.add_message(message).complete()

# Access conversation state
latest = response.latest_message
all_messages = response.messages
```

### Conversation Branching

The immutable design enables branching chats:

```python
# Build conversation context
base_chat = Chat(model=model)
    .add_message(SystemMessage("You are a helpful assistant."))
    .add_message(UserMessage("Tell me about renewable energy"))
    .complete()

# Branch into different explorations
branch_a = base_chat.add_message(UserMessage("Focus on solar power"))
branch_b = base_chat.add_message(UserMessage("Focus on wind power"))

solar_response = branch_a.complete()
wind_response = branch_b.complete()

# Original base_chat remains unchanged for further branching
```

### Message Types

```python
from patterpunk.llm.messages import SystemMessage, UserMessage, AssistantMessage

# System prompts
system_msg = SystemMessage("You are a helpful assistant.")

# User input with optional structured output
user_msg = UserMessage(
    "Analyze this data",
    structured_output=MyPydanticModel,  # Optional
    allow_tool_calls=True  # Default: True
)

# Assistant responses (created automatically by complete())
assistant_msg = AssistantMessage("The capital of France is Paris.")

# ToolCallMessage created automatically when LLM requests tools
```

### Provider Models

Each provider implements the same interface:

```python
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.models.bedrock import BedrockModel

# OpenAI
openai_model = OpenAiModel(model="gpt-4.1", temperature=0.7)

# Anthropic
anthropic_model = AnthropicModel(
    model="claude-sonnet-4-20250514", 
    temperature=0.7,
    max_tokens=1000
)

# AWS Bedrock
bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    temperature=0.1
)
```

## Provider Configuration

### OpenAI
Set API key via environment variable:
```bash
export PP_OPENAI_API_KEY="your-api-key"
```

### Anthropic
Set API key via environment variable:
```bash
export PP_ANTHROPIC_API_KEY="your-api-key"  
```

### AWS Bedrock
Configure AWS credentials:
```bash
export PP_AWS_ACCESS_KEY_ID="your-access-key"
export PP_AWS_SECRET_ACCESS_KEY="your-secret-key"
export PP_AWS_REGION="us-east-1"
```

### Google Vertex AI
```bash
export PP_GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
export PP_GEMINI_PROJECT="your-project-id"
```

### Ollama
```bash
export PP_OLLAMA_API_ENDPOINT="http://localhost:11434"
```

## Tool Calling

Convert Python functions with type hints into LLM tools:

```python
def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: 72°{unit[0].upper()}"

def calculate_tip(bill_amount: float, tip_percentage: float = 18.0) -> float:
    """Calculate tip amount."""
    return bill_amount * (tip_percentage / 100)

# Add tools to chat
chat = Chat(model=model).with_tools([get_weather, calculate_tip])

response = chat.add_message(
    UserMessage("What's the weather in Paris and calculate 18% tip on $50?")
).complete()
```

**Execution Flow**: LLM generates tool calls → Functions execute automatically → Results integrate back → LLM continues with context

## MCP Servers

Model Context Protocol integration for external tool execution:

```python
from patterpunk.lib.mcp import MCPServerConfig

# HTTP transport (remote servers)
weather_server = MCPServerConfig(
    name="weather-server",
    url="http://mcp-weather-server:8000/mcp"
)

# Stdio transport (local subprocesses)
filesystem_server = MCPServerConfig(
    name="filesystem",
    command=["python", "-m", "mcp_filesystem"],
    env={"MCP_FILESYSTEM_ROOT": "/workspace"}
)

# Integration with Chat
chat = Chat(model=model).with_mcp_servers([weather_server, filesystem_server])

# Combine with function tools
chat = Chat(model=model).with_tools([calculate_tip]).with_mcp_servers([weather_server])
```

## Structured Output

Use Pydantic models for structured LLM responses:

```python
from pydantic import BaseModel, Field
from typing import List

class WeatherReport(BaseModel):
    location: str = Field(description="City name")
    temperature: float = Field(description="Temperature in Fahrenheit")
    conditions: str = Field(description="Weather conditions")
    forecast: List[str] = Field(description="3-day forecast")

# Request structured output
response = chat.add_message(
    UserMessage(
        "Get weather report for Tokyo",
        structured_output=WeatherReport
    )
).complete()

# Access parsed Pydantic model
weather_data = response.parsed_output
print(f"Temperature: {weather_data.temperature}°F")
```

### Automatic Retry
If structured output parsing fails, patterpunk automatically retries with error feedback to the LLM.

## Advanced Features

### Message Templating
```python
user_msg = UserMessage("Hello {{name}}, today is {{date}}")
prepared_msg = user_msg.prepare({"name": "Alice", "date": "2024-01-15"})
```

### Model Switching
```python
fast_model = OpenAiModel(model="gpt-4.1-mini")
powerful_model = OpenAiModel(model="gpt-4.1")

response = chat.add_message(
    UserMessage("Simple question").set_model(fast_model)
).add_message(
    UserMessage("Complex analysis").set_model(powerful_model)  
).complete()
```

### JSON Extraction & Error Handling
```python
# Extract JSON from assistant messages
json_data = chat.extract_json()

# Handle structured output errors
from patterpunk.llm.chat import StructuredOutputParsingError
try:
    response = chat.complete()
    data = response.parsed_output
except StructuredOutputParsingError as e:
    print(f"Parsing failed: {e}")
```

## Agents

Agents provide reusable LLM workflow components with type-safe data flow and templated prompts:

```python
from dataclasses import dataclass
from typing import List
from patterpunk.llm.agent import Agent
from patterpunk.llm.chain import AgentChain, Parallel

@dataclass
class ReviewRequest:
    product_name: str
    price: float

@dataclass  
class ProductReview:
    rating: int
    summary: str

@dataclass
class ResearchInput:
    topic: str

@dataclass
class ResearchOutput:
    findings: str
    sources: List[str]

class ReviewAgent(Agent[ReviewRequest, ProductReview]):
    @property
    def model(self):
        return OpenAiModel(model="gpt-4.1")
    
    @property
    def _user_prompt_template(self) -> str:
        return "Review {{ product_name }} (${{ price }}). Rate 1-5 with summary."

class ResearchAgent(Agent[ResearchInput, ResearchOutput]):
    @property
    def model(self):
        return OpenAiModel(model="gpt-4.1")
    
    @property
    def _user_prompt_template(self) -> str:
        return "Research {{ topic }}. Provide findings and list 3-5 sources."

class SummaryAgent(Agent[ResearchOutput, str]):
    @property
    def model(self):
        return OpenAiModel(model="gpt-4.1-mini")
    
    @property
    def _user_prompt_template(self) -> str:
        return "Summarize these research findings in 2-3 sentences: {{ findings }}"

# Usage
agent = ReviewAgent()
review = agent.execute(ReviewRequest("Headphones Pro", 299.99))

# Chain agents sequentially
research_agent = ResearchAgent()
summary_agent = SummaryAgent()
chain = AgentChain([research_agent, summary_agent])

# Run agents in parallel  
parallel = Parallel([agent_a, agent_b, agent_c])
```
