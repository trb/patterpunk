# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Testing:**
- `pytest` - Run test suite from `/workspace/patterpunk/src/` directory
- **IMPORTANT: DO NOT execute tests unless explicitly instructed by the user**
- Tests are located in `patterpunk/src/tests/` with provider-specific test files

**Package Management:**
- `pip install -e .` - Install package in development mode (from `patterpunk/src/`)
- Dependencies: core (`requirements.txt`), testing (`test.requirements.txt`), build (`build.requirements.txt`)

**Code Formatting:**
- Uses `black` for formatting (available in test requirements)

## Architecture Overview

Patterpunk is an LLM provider abstraction library that provides a unified interface across OpenAI, Anthropic, AWS Bedrock, Google Vertex AI, and Ollama. The core philosophy is provider-agnostic development with all provider-specific functionality isolated in model implementations.

### Core Design Principles

1. **Chainable Immutable Interface**: All operations return new instances rather than modifying existing ones
   ```python
   answer = chat.add_message(system_message).add_message(user_message).complete().latest_message
   ```

2. **Provider Abstraction**: Provider-specific code is isolated in `/patterpunk/src/patterpunk/llm/models/` files. The Chat class and Message classes should never contain provider-specific code.

3. **Structured Output Support**: Uses Pydantic models with automatic fallback for providers without native structured output support.

### Key Components

- **Chat Class** (`llm/chat.py`): Main entrypoint with conversation state management and chainable interface
- **Message Types** (`llm/messages.py`): SystemMessage, UserMessage, AssistantMessage, ToolCallMessage with Jinja2 templating support
- **Agent System** (`llm/agent.py`, `llm/chain.py`): Workflow abstractions with AgentChain for sequential execution and Parallel for concurrent execution
- **Provider Models** (`llm/models/`): Each provider implements abstract Model base class with provider-specific API handling

### File Structure
```
patterpunk/src/patterpunk/
├── llm/
│   ├── chat.py          # Main Chat class entrypoint
│   ├── messages.py      # Message type definitions
│   ├── agent.py         # Agent workflow abstraction
│   ├── chain.py         # Agent chains and parallel execution
│   └── models/          # Provider implementations
│       ├── base.py      # Abstract Model base class
│       ├── openai.py    # OpenAI provider
│       ├── anthropic.py # Anthropic provider
│       ├── bedrock.py   # AWS Bedrock provider
│       ├── google.py    # Google Vertex AI provider
│       └── ollama.py    # Ollama provider
├── lib/                 # Utility libraries
│   ├── extract_json.py  # JSON extraction utilities
│   └── structured_output.py # Structured output handling
├── config.py            # Configuration and provider setup
└── logger.py            # Logging utilities
```

## Code Style Requirements

**CRITICAL RULES:**
- **NEVER ADD COMMENTS** - Code must be self-documenting
- **DO NOT CREATE documentation files** unless explicitly requested
- Use functional programming principles with immutability
- Minimize object-oriented programming and inheritance
- Never abbreviate variable names (use `signature` not `sig`)
- Maintain chainable immutable interface patterns
- Keep provider-specific code isolated in model files only

**Functional Programming Guidelines:**
- Prefer immutable data structures and pure functions
- Use list comprehensions over explicit loops
- Avoid complex lambda expressions
- Default to functions and modules over classes
- Use classes only for essential state management or abstractions

**File Organization:**
- One primary export per file
- Deep module nesting with focused responsibilities  
- Separate files for distinct domain concepts
- Keep helper functions with their primary function when only used locally

## Configuration

All settings use environment variables with `PP_` prefix:
- Provider credentials: `PP_OPENAI_API_KEY`, `PP_ANTHROPIC_API_KEY`, etc.
- Model defaults: `PP_DEFAULT_MODEL`, `PP_DEFAULT_TEMPERATURE`
- Regional settings: `PP_AWS_REGION`, `PP_GOOGLE_LOCATION`

## Research Guidelines

When performing web research:
- Issue limited number of highly divergent search queries
- Avoid multiple searches with slight wording variations
- Search for very different terms to maximize coverage

## Supported Providers

- **OpenAI**: GPT models with structured output, reasoning effort support
- **Anthropic**: Claude models with temperature, top-p, top-k controls  
- **AWS Bedrock**: Multiple model families (Claude, Llama, Mistral, Titan)
- **Google**: Vertex AI integration
- **Ollama**: Local model serving