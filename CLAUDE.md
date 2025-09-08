# Patterpunk Coding Directions

This file provides guidance to Claude Code (claude.ai/code) when working on patterpunk.

## Additional Information

You MUST access the following files if you work on related features of patterpunk:

- './AGENTS.md': Principles and design choices and goals for the agentic features in patterpunk, like the Agent base class or the agent chain functionality
- './MULTIMODAL.md': Patterpunk supports multi-modal input like files, images, audio, video, etc - this file include information about how patterpunk handles multi-model content
- './PROMPT_CACHING.md': Some providers support prompt caching for performance/cost-reduction reasons, this file details how patterpunk supports prompt caching
- './TOOL_CALLING.md': Documentation on how patterpunk implements tool calling
- './README.md': Developer-oriented documentation on how to use patterpunk, helpful to understand how patterpunk is intended to be used

## Development Commands
- Tests are located in `patterpunk/src/tests/` with provider-specific test files
- Run tests with `docker compose run --entrypoint '/bin/bash -c' patterpunk '/app/bin/test.dev $TEST'` -  the single-quotes around `/app/bin/test.dev $TEST` are paramount, without them ALL tests will be excuted. It's a FATAL violation to run the tests without wrapping the command and test case in single-quotes!
- Test files should be given relative to `patterpunk/src` prefixed with `/app` as that is the mount point in the container. E.g. for anthropic tests, the path would be `/app/tests/test_anthropic.py`
- Dependencies: core (`requirements.txt`), testing (`test.requirements.txt`), build (`build.requirements.txt`)

**Code Formatting:**
- Run `docker compose run --entrypoint '/bin/bash -c' --remove-orphans patterpunk black` for formatting

## General Rules

1. You MUST NOT estimate time periods. Under no circumstances should you add time estimates, to any file, in any context.


<codeDesignRules importance="FATAL" mustFollow="true">
YOU MUST FOLLOW THESE CODE DESIGN RULES! IT IS A FATAL VIOLATION NOT TO FOLLOW THESE RULES!

## Code Design Rules - How to write code in patterunk!
- FATAL: DO NOT ADD SINGLE-LINE COMMENTS, UNDER ANY CIRCUMSTANCES!
- FATAL: NEVER ADD LOGIC OR EXPORTS TO `__init__.py` FILES! EVER! ALL `__init__.py` FILES MUST REMAIN EMPTY!
- You MUST NOT mock any interface when writing tests unless VERY EXPLICTLY instructed to do so by the user. Our tests are all integration tests and should hit all services.
- Code MUST be expressive and easy to understand. Avoid complex code structures, long if-statements, long functions, etc - prefer composition instead of complex code structures
- **Incremental progress over big bangs** - Small changes that compile
- **Learning from existing code** - Study and plan before implementing
- **Pragmatic over dogmatic** - Adapt to project reality
- **Clear intent over clever code** - Be boring and obvious
- Single responsibility per function/class
- Avoid premature abstractions
- No clever tricks - choose the boring solution
- If you need to explain it, it's too complex
- Split long functions into smaller ones
- Keep all code structures - functions, classes, modules, etc. - small and easy to understand
- Compose instead of inline
- Single-use functions still have value as they convey context that otherwise would be missing, and they separate functionality into smaller chunks
- **MUST REMEMBER:** Design code for humans, whose working memory can handle 5 memory chunks at most, where a chunk is a new piece of information, like an unknown variable, function interface, etc., keep those under 5 chunks
- Code must be self-documenting through clear variable names and structure
- Never abbreviate variable names (use `signature` not `sig`)
- Keep provider-specific code isolated in model files only
- NO docstrings (unless absolutely required by a framework)
- NO inline comments explaining what code does
- NO comment blocks
- NO TODO comments
</codeDesignRules>

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

<criticalResearchInstructions>
## Critical instructions for research tools!

- **TOOLS**: `mcp__research__[find|perform]_[simple|deep]_*`
- FIND before performing a new research for both simple and deep research
- FIRST TRY TO FIND A REPORT!
- Research requests need to be detailed and in-depth, providing all relevant context for the research. These aren't simple search queries!
- State clearly what you want the research report to address, make it abundantly clear and completely unambiguous what the topics of the report should be
- **DECISION FRAMEWORK** Simple vs. Deep research: Simple research is faster (about 5 minutes) and is a great solution for specific research like how to use a library, how to interface with a specific AWS service, how to write good integration tests. Deep research takes a long time (about 30 minutes, set your timeouts correctly), and will iteratively explore the subject matter. Use this for broad research topics, for example how to implement a specific cloud architecture covering many services, how to prompt LLMs, how to fine-tune an LLM - broad subject matters that cover a lot of knowledge space.
- Deep research takes about 30 minutes, you MUST handle such a long tool call!
- FIND deep research freely (retrieving a report is fast and inexpensive) but consider PERFORMING a report carefully (it takes a long time and is expensive)
</criticalResearchInstructions>

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
