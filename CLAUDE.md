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
- Run tests with `docker compose -p patterpunk run --rm patterpunk -c '/app/bin/test.dev $TEST'` - the single-quotes around the command are paramount, without them ALL tests will be executed. It's a FATAL violation to run tests without wrapping the command in single-quotes!
- Test files should be given relative to `patterpunk/src` prefixed with `/app` as that is the mount point in the container. E.g. for anthropic tests, the path would be `/app/tests/test_anthropic.py`
- Dependencies: core (`requirements.txt`), testing (`test.requirements.txt`), build (`build.requirements.txt`)

**Code Formatting:**
- Run `docker compose -p patterpunk run --rm patterpunk -c 'black /app/patterpunk'` for formatting

**Note:** The `-p patterpunk` flag is required because the patterpunk Docker Compose stack runs as a separate project. This ensures commands target the correct container stack regardless of the current working directory.

## General Rules

1. You MUST NOT estimate time periods. Under no circumstances should you add time estimates, to any file, in any context.


<code-design-rules importance="FATAL" mustFollowEveryTime="true">

## Code Design Rules

You MUST follow these rules when writing code. They'll guide you to well-structured, clean and maintainable code! It's of utmost, FATAL importance that you follow these rules.

**FUNDAMENTAL CONSTRAINT**
- Working memory holds ~4 chunks of information; exceeding this causes comprehension failure
- Every abstraction, indirection, and context switch consumes a chunk
- Optimize for minimal mental effort to understand and modify code

**FUNCTION/METHOD DESIGN**
- Extract complex conditionals into descriptively-named boolean variables (e.g., `isEligibleForDiscount` instead of `user.age > 18 && user.memberSince < 2020 && !user.suspended`)
- Use early returns to avoid nested conditionals - handle edge cases first, then focus on happy path
- One abstraction level per function - don't mix high-level orchestration with low-level details
- Deep modules preferred: simple interface hiding complex implementation (e.g., `database.save(user)` hiding connection pooling, retries, transactions)
- Pure functions better than stateful objects - push side effects to system edges

**ABSTRACTION GUIDELINES**
- "A little copying is better than a little dependency" - prefer small code duplication over wrong abstraction
- Build abstractions only after pattern emerges 3+ times, not speculatively
- Always provide escape hatches - allow bypassing abstraction for edge cases
- Abstract for changeability not reusability - focus on what varies in YOUR domain
- Prefer composition over inheritance - inheritance creates tight coupling and mental stack overflow

**CODE ORGANIZATION**
- Organize by feature/domain not technical layer (e.g., `user/` not `controllers/, models/, views/`)
- Colocate related logic - group code that changes together
- Keep files/modules focused - split when cognitive load exceeds ~4 concepts
- Minimize cross-module dependencies - each module should be understandable in isolation
- Single source of truth - avoid duplicate state/logic

**NAMING & CLARITY**
- Intent-revealing names better than implementation-describing (e.g., `validateUserEligibility()` not `checkUserObject()`)
- Explicit better than implicit - no hidden magic or assumed knowledge
- Self-documenting code better than comments - if you need to explain it, simplify it
- Consistent naming conventions throughout codebase - reduce mental translation

**ARCHITECTURAL DECISIONS**
- Monolith-first - microservices add distributed complexity; split only when proven necessary
- Framework-agnostic core logic - frameworks at edges only (controllers, adapters)
- Push complexity to edges - keep core business logic simple, handle messiness in adapters
- Bounded contexts better than arbitrary boundaries - split by domain not technical concerns

**AVOID THESE PATTERNS**
- Deep inheritance hierarchies (more than 2 levels)
- Lasagna architecture (too many layers of indirection)
- Mixed abstraction levels in same scope
- Shared mutable state across components
- Temporal coupling (hidden order dependencies)
- Action-at-distance (modifying state in unexpected places)
- Clever/elegant code - boring and obvious wins
- Speculative generality - YAGNI (You Aren't Gonna Need It)
- Framework religion - use frameworks pragmatically not dogmatically
- DRY extremism - some duplication acceptable to reduce coupling

**STATE & DATA FLOW**
- Immutability by default - mutate only when performance requires
- Make illegal states unrepresentable in type system
- Explicit data flow - avoid hidden channels
- Fail fast and loud - surface errors immediately
- Use types as documentation - rich domain types over primitives

**REFACTORING PRINCIPLES**
- Refactor when pattern is clear, not when suspected
- Optimize for deletion - make code easy to remove
- Leave breadcrumbs - document non-obvious decisions
- Business logic is inherently messy - embrace it, don't over-abstract
- Code for median developer competence not genius level

**COMPLEXITY MANAGEMENT**
- Each function should require 4 or fewer mental chunks to understand
- Flatten deeply nested structures - use guard clauses, early returns
- Avoid callback hell - use async/await or similar patterns
- Minimize action distance - keep cause and effect close
- Reduce state space - fewer possible states equals fewer bugs
- Hide complexity that won't change, expose what will vary

**INTERFACE DESIGN**
- Minimal surface area - fewer public methods/properties
- Consistent patterns across similar interfaces
- Progressive disclosure - simple common cases, complex rare cases
- Avoid boolean parameters - use enums or separate methods
- Return consistent types - avoid sometimes-null, sometimes-array confusion

**PRACTICAL SIMPLIFICATION**
- If explaining takes more than 2 sentences, code is too complex
- If you need a diagram to understand flow, refactor
- If new developers consistently misunderstand, clarify
- If changes require touching many files, reconsider boundaries
- If testing is hard, design is probably wrong

</code-design-rules>

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
