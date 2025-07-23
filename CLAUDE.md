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

Patterpunk is an LLM provider abstraction library that provides a unified interface across OpenAI, Anthropic, AWS Bedrock, Google Vertex AI, and Ollama. The core philosophy is provider-agnostic development with all provider-specific functionality isolated in model implementations. The library includes comprehensive tool calling support and MCP (Model Context Protocol) integration for external tool execution.

### Core Design Principles

1. **Chainable Immutable Interface**: All operations return new instances rather than modifying existing ones
   ```python
   answer = chat.add_message(system_message).add_message(user_message).complete().latest_message
   ```

2. **Provider Abstraction**: Provider-specific code is isolated in `/patterpunk/src/patterpunk/llm/models/` files. The Chat class and Message classes should never contain provider-specific code.

3. **Structured Output Support**: Uses Pydantic models with automatic fallback for providers without native structured output support.

4. **Tool Calling Integration**: Seamless function calling with automatic conversion from Python functions to OpenAI-compatible tool definitions. Supports both direct function tools and MCP (Model Context Protocol) servers.

5. **MCP Protocol Support**: Full integration with Model Context Protocol for external tool execution via HTTP and stdio transports with automatic tool discovery and execution.

### Key Components

- **Chat Class** (`llm/chat.py`): Main entrypoint with conversation state management, chainable interface, and tool calling via `with_tools()` and `with_mcp_servers()` methods
- **Message Types** (`llm/messages.py`): SystemMessage, UserMessage, AssistantMessage, ToolCallMessage with Jinja2 templating support and structured tool call data
- **Tool Calling System** (`llm/types.py`, `lib/function_to_tool.py`): Automatic conversion of Python functions to OpenAI-compatible tool definitions with type introspection and docstring parsing
- **MCP Integration** (`lib/mcp/`): Complete Model Context Protocol implementation with HTTP/stdio transports, tool discovery, and execution
- **Agent System** (`llm/agent.py`, `llm/chain.py`): Workflow abstractions with AgentChain for sequential execution and Parallel for concurrent execution
- **Provider Models** (`llm/models/`): Each provider implements abstract Model base class with provider-specific API handling and tool calling support

### File Structure
```
patterpunk/src/patterpunk/
├── llm/
│   ├── chat.py          # Main Chat class entrypoint with tool calling
│   ├── messages.py      # Message type definitions including ToolCallMessage
│   ├── types.py         # Tool calling type definitions and interfaces
│   ├── agent.py         # Agent workflow abstraction
│   ├── chain.py         # Agent chains and parallel execution
│   └── models/          # Provider implementations
│       ├── base.py      # Abstract Model base class
│       ├── openai.py    # OpenAI provider with native tool calling
│       ├── anthropic.py # Anthropic provider with tool use support
│       ├── bedrock.py   # AWS Bedrock provider with tool calling
│       ├── google.py    # Google Vertex AI provider with tool calling
│       └── ollama.py    # Ollama provider with tool calling
├── lib/                 # Utility libraries
│   ├── extract_json.py  # JSON extraction utilities
│   ├── structured_output.py # Structured output handling
│   ├── function_to_tool.py # Python function to OpenAI tool conversion
│   └── mcp/             # Model Context Protocol implementation
│       ├── client.py    # MCP client with HTTP/stdio transports
│       ├── server_config.py # MCP server configuration
│       ├── tool_converter.py # MCP to OpenAI tool format conversion
│       └── exceptions.py # MCP-specific exception classes
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

### MCP Server Configuration

MCP servers are configured via `MCPServerConfig` objects supporting dual transport modes:

**HTTP Transport** (for remote/containerized servers):
```python
from patterpunk.lib.mcp import MCPServerConfig

weather_server = MCPServerConfig(
    name="weather-server",
    url="http://mcp-weather-server:8000/mcp"
)
```

**Stdio Transport** (for local subprocess servers):
```python
filesystem_server = MCPServerConfig(
    name="filesystem",
    command=["python", "-m", "mcp_filesystem"],
    env={"MCP_FILESYSTEM_ROOT": "/workspace"}
)
```

MCP integration requires optional dependencies:
- `requests` for HTTP transport
- Subprocess servers run as separate processes

## Research Guidelines

When performing web research:
- Issue limited number of highly divergent search queries
- Avoid multiple searches with slight wording variations
- Search for very different terms to maximize coverage

## Supported Providers

- **OpenAI**: GPT models with native tool calling, structured output, reasoning effort support
- **Anthropic**: Claude models with tool use support, temperature, top-p, top-k controls  
- **AWS Bedrock**: Multiple model families (Claude, Llama, Mistral, Titan) with tool calling support
- **Google**: Vertex AI integration with tool calling capabilities
- **Ollama**: Local model serving with tool calling support

## Tool Calling Usage

**Function-based Tools**:
```python
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}"

chat = Chat().with_tools([get_weather])
```

**MCP Server Integration**:
```python
from patterpunk.lib.mcp import MCPServerConfig

weather_mcp = MCPServerConfig(name="weather", url="http://weather-server:8000/mcp")
chat = Chat().with_mcp_servers([weather_mcp]).with_tools([get_weather])

response = chat.add_message(UserMessage("What's the weather?")).complete()
```

Tools are automatically executed after LLM completion when tool calls are detected.