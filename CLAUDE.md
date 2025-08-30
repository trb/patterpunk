# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Additional Information

You MUST access the following files if you work on related features of patterpunk:

- './AGENTS.md': Principles and design choices and goals for the agentic features in patterpunk, like the Agent base class or the agent chain functionality
- './MULTIMODAL.md': Patterpunk supports multi-modal input like files, images, audio, video, etc - this file include information about how patterpunk handles multi-model content
- './PROMPT_CACHING.md': Some providers support prompt caching for performance/cost-reduction reasons, this file details how patterpunk supports prompt caching
- './TOOL_CALLING.md': Documentation on how patterpunk implements tool calling
- './README.md': Developer-oriented documentation on how to use patterpunk, helpful to understand how patterpunk is intended to be used

## Reviews

You MUST review significant changes with the review council members! Each review council member is implement as a sub-agent that you should call as a task. You should call all members in parallel.

<reviewInstructions>

Initiate a review by creating a journal as the council-reviews/$REVIEW_NAME.md` file. Append to the journal everything related to the review - what you called the sub-agents with, their responses, iterations you performed, and the final decision.

List of members:

<councilMembers implementedAsSubAgents="true" callSubagents="true">
You have access to the following council members. The council members are implemented as subagents. You must call these subagents when it's their turn to review the information.

Each subagent has its own context. You have to provide all relevant information and context each time you call the subagent. It will not remember previous rounds!

For each round of discussion you must call each subagent!

<creativeCouncilMember>
The creative council member is responsible for reviewing the creative aspects of the decision. It prefers novel and progressive solutions. The subagent is called `review-council-the-creative-member`.
</creativeCouncilMember>
<conservativeCouncilMember>
The conservative council member is responsible for reviewing the conservative aspects of the decision. It prefers solutions that are well-established and well-understood. The subagent is called `review-council-the-conservative-member`.
</conservativeCouncilMember>
</councilMembers>

Follow this review process:

<reviewProcess>
<step id="1">
Call each sub-agent with the provided information and augmented information, instructing them to review the information and to provide a meaningful review. Do this in parallel. Add each response to the document. You MUST call the subagent as a tool call!
</step>
<step id="2">
At least once, pass the provided information, the augmented information and the responses to each sub-agent, so they can respond to one another.
</step>
<step id="3">
Append all responses to the logging document.
Decide if another round of review would be useful. If so, repeat the process from step 1. Justify your decision and log it in the logging document.
</step>
<finalStep>
Once you are satisfied that all sub-agents have provided meaningful feedback and further discussion is not necessary, compile the final report. Record this in the journal.
</finalStep>
</reviewProcess>

</reviewInstructions>

## Development Commands
- Tests are located in `patterpunk/src/tests/` with provider-specific test files
- Run tests with `docker compose run --entrypoint '/bin/bash -c' patterpunk /app/bin/test.dev $TEST_FILE`
- Test files should be given relative to `patterpunk/src` prefixed with `/app` as that is the mount point in the container. E.g. for anthropic tests, the path would be `/app/tests/test_anthropic.py`
- Dependencies: core (`requirements.txt`), testing (`test.requirements.txt`), build (`build.requirements.txt`)

**Code Formatting:**
- Uses `black` for formatting

## General Rules

1. You MUST NOT estimate time periods. Under no circumstances should you add time estimates, to any file, in any context.

## Coding Rules

1. You MUST NOT mock any interface when writing tests unless VERY EXPLICTLY instructed to do so by the user. Our tests are all integration tests and should hit all services.
2. Code MUST be expressive and easy to understand. Avoid complex code structures, long if-statements, long functions, etc - prefer composition instead of complex code structures
3. Keep human working memory limitations in mind, humans can typically keep 3–5 chunks in their working memory. "Chunk" is a new piece of information—an unfamiliar variable, a new function interface, etc - aim to write code that requires less than 5 chunks to understand

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

6. **Reasoning Controls**: ThinkingConfig integration for providers supporting reasoning modes (OpenAI o3-mini, Anthropic Claude, Google Gemini) with effort levels and token budgets.

7. **Multimodal Content Handling**: Unified interface for images, files, audio, and video content through MultimodalChunk with support for various sources (files, URLs, base64, data URIs, bytes).

8. **Content Chunking System**: Polymorphic content support with TextChunk, CacheChunk, and MultimodalChunk for fine-grained control over message content and optimization.

### Key Components

- **Chat Class** (`llm/chat/core.py`): Core conversation management with chainable interface, delegating specialized operations to focused modules via `with_tools()` and `with_mcp_servers()` methods
- **Message Types** (`llm/messages/`): Split into focused modules - SystemMessage, UserMessage, AssistantMessage, ToolCallMessage with Jinja2 templating support and structured tool call data
- **Content Chunks** (`llm/cache.py`, `llm/text.py`, `llm/multimodal.py`): Polymorphic content system with CacheChunk for cacheable content with TTL, TextChunk for plain text, and MultimodalChunk for media files
- **Reasoning Controls** (`llm/thinking.py`): ThinkingConfig class for controlling reasoning effort levels and token budgets across supported providers
- **Tool Calling System** (`llm/types.py`, `lib/function_to_tool/`): Enhanced conversion of Python functions to OpenAI-compatible tool definitions with advanced docstring parsing and type introspection
- **MCP Integration** (`lib/mcp/`): Complete Model Context Protocol implementation with HTTP/stdio transports, tool discovery, and execution
- **Agent System** (`llm/agent.py`, `llm/chain.py`): Workflow abstractions with AgentChain for sequential execution and Parallel for concurrent execution
- **Provider Models** (`llm/models/`): Each provider implements abstract Model base class with provider-specific API handling, tool calling support, and reasoning integration

### File Structure
```
patterpunk/src/patterpunk/
├── llm/
│   ├── chat.py          # Main Chat class entrypoint (imports from chat/)
│   ├── chat/            # Chat functionality modules
│   │   ├── core.py      # Core Chat class with conversation management
│   │   ├── tools.py     # Tool calling and MCP server configuration
│   │   ├── structured_output.py # Structured output parsing with retry
│   │   └── exceptions.py # Chat-specific exceptions
│   ├── messages/        # Message type definitions (split by type)
│   │   ├── base.py      # Base Message class
│   │   ├── system.py    # SystemMessage class
│   │   ├── user.py      # UserMessage class
│   │   ├── assistant.py # AssistantMessage class
│   │   ├── tool_call.py # ToolCallMessage class
│   │   ├── templating.py # Jinja2 templating support
│   │   ├── cache.py     # Cache-related message utilities
│   │   ├── structured_output.py # Structured output message handling
│   │   ├── roles.py     # Message role constants
│   │   └── exceptions.py # Message-specific exceptions
│   ├── cache.py         # CacheChunk class for cacheable content
│   ├── text.py          # TextChunk class for plain text content
│   ├── multimodal.py    # MultimodalChunk class for media content
│   ├── thinking.py      # ThinkingConfig for reasoning controls
│   ├── types.py         # Tool calling type definitions and interfaces
│   ├── tool_types.py    # Additional tool type definitions
│   ├── agent.py         # Agent workflow abstraction
│   ├── chain.py         # Agent chains and parallel execution
│   ├── defaults.py      # Default configuration values
│   └── models/          # Provider implementations
│       ├── base.py      # Abstract Model base class
│       ├── openai.py    # OpenAI provider with native tool calling
│       ├── anthropic.py # Anthropic provider with tool use support
│       ├── bedrock.py   # AWS Bedrock provider with tool calling
│       ├── google.py    # Google Vertex AI provider with tool calling
│       └── ollama.py    # Ollama provider with tool calling
├── config/              # Configuration modules
│   ├── defaults.py      # Default configuration values
│   └── providers/       # Provider-specific configurations
│       ├── openai.py    # OpenAI configuration
│       ├── anthropic.py # Anthropic configuration
│       ├── bedrock.py   # AWS Bedrock configuration
│       ├── google.py    # Google Vertex AI configuration
│       └── ollama.py    # Ollama configuration
├── lib/                 # Utility libraries
│   ├── extract_json.py  # JSON extraction utilities
│   ├── structured_output.py # Structured output handling
│   ├── function_to_tool/ # Python function to OpenAI tool conversion
│   │   ├── converter.py # Main conversion logic
│   │   ├── inspection.py # Function signature analysis
│   │   ├── schema.py    # Schema generation utilities
│   │   ├── cleaning.py  # Description cleaning utilities
│   │   └── docstring/   # Docstring parsing modules
│   │       ├── parser.py # Main docstring parser
│   │       ├── simple.py # Regex-based parsing
│   │       └── advanced.py # External parser integration
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
- **ABSOLUTELY NO COMMENTS IN CODE** - This means:
  - NO docstrings (unless absolutely required by a framework)
  - NO inline comments explaining what code does
  - NO comment blocks
  - NO TODO comments
  - NO type hint comments
  - NO disabled code comments
  - The ONLY exception is copyright headers if required
  - Code must be self-documenting through clear variable names and structure
- **DO NOT CREATE documentation files** unless explicitly requested
- **DO NOT CREATE example files** unless explicitly requested
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

**Module Organization:**
- Split large modules into focused submodules (e.g., `chat/` and `messages/` directories)
- One primary export per file with clear domain boundaries
- Deep module nesting with focused responsibilities  
- Separate files for distinct domain concepts (e.g., separate files for each message type)
- Keep helper functions with their primary function when only used locally

**Content Chunk Patterns:**
- Use polymorphic content with TextChunk, CacheChunk, and MultimodalChunk for flexible message construction
- Prefer content lists over simple strings when fine-grained control is needed
- Implement consistent interfaces across chunk types (e.g., similar constructor patterns and repr methods)
- Design chunk types to be provider-agnostic with provider-specific handling in model implementations

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

## Research and Websearch Guidelines

When performing web research:
- Issue multiple highly divergent and contrasting search queries
- Avoid multiple searches with slight wording variations
- Search for very different terms to maximize coverage

## Supported Providers

- **OpenAI**: GPT models with native tool calling, structured output, reasoning effort support
- **Anthropic**: Claude models with tool use support, temperature, top-p, top-k controls  
- **AWS Bedrock**: Multiple model families (Claude, Llama, Mistral, Titan) with tool calling support
- **Google**: Vertex AI integration with tool calling capabilities
- **Ollama**: Local model serving with tool calling support
