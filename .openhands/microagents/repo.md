# Patterpunk Repository Guide

## Purpose

Patterpunk is an LLM-interfacing library that provides abstractions to make it easy to work with multiple different LLM providers. The library abstracts away provider-specific APIs and functionality differences, allowing developers to use a unified interface regardless of whether they're working with OpenAI, Anthropic, AWS Bedrock, Google, or Ollama.

The core philosophy is to handle all provider idiosyncrasies internally, so users don't need to consider which specific provider they're using beyond being aware of what functionality each provider/model supports.

## General Setup

### Development Environment
- **Docker-based development**: The repository uses `docker-compose.yml` for building and running tests
- **Python package**: The library is packaged using setuptools with version 0.0.19
- **Minimal dependencies**: Core dependency is only Jinja2 for templating
- **Environment-based configuration**: Provider credentials and settings are configured via environment variables with `PP_` prefix

### Supported Providers
- **OpenAI**: GPT models with support for structured output, reasoning effort, and various parameters
- **Anthropic**: Claude models with temperature, top-p, top-k controls
- **AWS Bedrock**: Multiple model families (Claude, Llama, Mistral, Titan)
- **Google**: Vertex AI integration
- **Ollama**: Local model serving

## Repository Structure

```
/workspace/
├── bin/                          # Helper scripts (usually don't touch unless instructed)
│   ├── commands/
│   ├── fix-permissions
│   ├── idea
│   └── pp
├── patterpunk/                   # Main library directory
│   ├── Dockerfile               # For testing/development only
│   └── src/                     # Library source code
│       ├── patterpunk/          # Core library code
│       │   ├── llm/             # LLM interaction components
│       │   │   ├── chat.py      # Main entrypoint - core Chat class
│       │   │   ├── messages.py  # Message types (System, User, Assistant, etc.)
│       │   │   ├── agent.py     # Agent abstraction for workflows
│       │   │   ├── chain.py     # Agent chains and parallel execution
│       │   │   └── models/      # Provider-specific implementations
│       │   │       ├── base.py      # Abstract base Model class
│       │   │       ├── openai.py    # OpenAI provider
│       │   │       ├── anthropic.py # Anthropic provider
│       │   │       ├── bedrock.py   # AWS Bedrock provider
│       │   │       ├── google.py    # Google Vertex AI provider
│       │   │       └── ollama.py    # Ollama provider
│       │   ├── lib/             # Utility libraries
│       │   │   ├── extract_json.py     # JSON extraction utilities
│       │   │   └── structured_output.py # Structured output handling
│       │   ├── config.py        # Configuration and provider setup
│       │   └── logger.py        # Logging utilities
│       ├── tests/               # Test suite
│       │   ├── test_*.py        # Provider-specific tests
│       │   └── conftest.py      # Test configuration
│       ├── setup.py             # Package configuration
│       └── requirements.txt     # Dependencies
├── docker-compose.yml           # For testing/development only
├── example.py                   # Basic usage example with Agent
├── chain_example.py             # Advanced example with AgentChain
└── README.md                    # Basic project description
```

## Core Design Principles

### 1. Chainable Interface
The Chat class uses a chainable, immutable interface where every method returns a deep-copied instance:

```python
answer = chat.add_message(system_message).add_message(user_message).complete().latest_message
```

This design enables:
- **Branching conversations**: Create multiple response variations from the same point
- **Immutability**: No risk of affecting other chat instances
- **Fluent API**: Natural, readable code flow

Example of branching:
```python
chat = chat.add_message(user_message)
[answer_1, answer_2, answer_3] = [chat.complete(), chat.complete(), chat.complete()]
```

### 2. Provider Abstraction
- **Model files** (`patterpunk/src/patterpunk/llm/models/`) contain ONLY provider-specific functionality
- **Chat class and Message classes** should NEVER contain provider-specific code
- Each provider maps patterpunk functionality to their specific offerings (e.g., structured output handling)

### 3. Structured Output Support
- Uses Pydantic models for type-safe structured responses
- Automatic fallback for providers without native structured output support
- Built-in retry logic with error correction prompts

## Key Components

### Chat Class (`llm/chat.py`)
- **Main entrypoint** for library usage
- Manages conversation state and message history
- Handles completion requests through provider models
- Supports structured output parsing with automatic retries
- Provides JSON extraction utilities

### Message Types (`llm/messages.py`)
- `SystemMessage`: System prompts and instructions
- `UserMessage`: User inputs, can include structured output requests
- `AssistantMessage`: Model responses
- `ToolCallMessage`: Function/tool call messages
- Support for Jinja2 templating in message content

### Agent System (`llm/agent.py`, `llm/chain.py`)
- **Agent**: Generic workflow abstraction with input/output types
- **AgentChain**: Sequential execution of agents
- **Parallel**: Concurrent execution of multiple agents
- Built-in structured output support using Python type hints

### Provider Models (`llm/models/`)
- Each provider implements the abstract `Model` base class
- Handle provider-specific API calls and parameter mapping
- Abstract away differences in structured output support
- Manage authentication and configuration per provider

## Development Guidelines

### Code Quality
- **Minimal changes**: Focus on solving the problem with minimal modifications
- **Clean code**: Efficient implementation with minimal comments
- **Provider separation**: Keep provider-specific code isolated in model files
- **Chainable design**: Maintain immutable, chainable interface patterns

### Testing
- Tests located in `patterpunk/src/tests/`
- Provider-specific test files for each supported LLM
- Examples demonstrate both basic usage and advanced patterns
- **Note**: Test environment setup not currently available in this workspace

### Configuration
- Environment variables with `PP_` prefix for all settings
- Provider credentials: `PP_OPENAI_API_KEY`, `PP_ANTHROPIC_API_KEY`, etc.
- Model defaults: `PP_DEFAULT_MODEL`, `PP_DEFAULT_TEMPERATURE`, etc.
- Regional settings: `PP_AWS_REGION`, `PP_GOOGLE_LOCATION`, etc.

## CI/CD Information

**No GitHub workflows found** - This repository does not currently have `.github/workflows/` directory or CI/CD automation configured.

## Usage Examples

### Basic Chat Usage
```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage, SystemMessage
from patterpunk.llm.models.openai import OpenAIModel

chat = Chat(model=OpenAIModel(model="gpt-4"))
response = chat.add_message(SystemMessage("You are a helpful assistant")).add_message(UserMessage("Hello!")).complete()
print(response.latest_message.content)
```

### Agent-based Workflow
```python
from patterpunk import Agent
from patterpunk.llm.models.openai import OpenAIModel

class MyAgent(Agent[InputType, OutputType]):
    @property
    def model(self):
        return OpenAIModel(model="gpt-4")
    
    @property
    def system_prompt(self):
        return "You are an expert assistant."
    
    @property
    def _user_prompt_template(self):
        return "{{ user_input }}"

agent = MyAgent()
result = agent.execute(input_data)
```

This repository follows a clean architecture pattern that separates concerns between conversation management, provider abstraction, and high-level workflow orchestration.