# MCP (Model Context Protocol) Integration

This module provides support for integrating MCP servers with patterpunk, allowing LLMs to call tools provided by external MCP servers.

## Installation

To use MCP functionality, install patterpunk with the MCP extra:

```bash
pip install patterpunk[mcp]
```

This installs the required `requests` dependency for HTTP transport.

## Basic Usage

### HTTP Transport (Recommended for containers)

```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage
from patterpunk.lib.mcp import MCPServerConfig

# Configure MCP servers
weather_server = MCPServerConfig(
    name="weather-server",
    url="http://mcp-weather-server:8000/mcp",  # External MCP server URL
    timeout=30.0
)

database_server = MCPServerConfig(
    name="database-server", 
    url="https://api.example.com/mcp",
    timeout=60.0
)

# Use with Chat - integrates seamlessly with existing tools
chat = (Chat()
    .with_mcp_servers([weather_server, database_server])
    .add_message(UserMessage("What's the weather in New York City?"))
)

# MCP tools are automatically executed when called by the LLM
response = chat.complete()
print(response.latest_message.content)
```

### Stdio Transport (For local processes)

```python
from patterpunk.lib.mcp import MCPServerConfig

# Configure local MCP server
local_server = MCPServerConfig(
    name="local-tools",
    command=["python", "/path/to/mcp_server.py"],
    env={"PYTHONPATH": "/custom/path"}
)

chat = Chat().with_mcp_servers([local_server])
```

### Combining MCP Servers with Regular Tools

```python
def local_function(message: str) -> str:
    '''A regular patterpunk tool function.'''
    return f"Local response: {message}"

chat = (Chat()
    .with_tools([local_function])  # Regular tools
    .with_mcp_servers([weather_server])  # MCP servers
    .add_message(UserMessage("Use both local and remote tools"))
)

response = chat.complete()
```

## Configuration

### MCPServerConfig Options

- `name`: Unique identifier for the server
- `url`: HTTP endpoint for MCP server (HTTP transport)
- `command`: Command to launch server subprocess (stdio transport)  
- `env`: Environment variables for subprocess (stdio only)
- `timeout`: Request timeout in seconds (default: 30.0)

### Transport Types

**HTTP Transport (`url` specified):**
- Used for remote MCP servers
- Ideal for containerized environments
- Uses Streamable HTTP protocol
- Supports session management

**Stdio Transport (`command` specified):**
- Used for local MCP server processes
- Server runs as subprocess
- Communication over stdin/stdout
- Automatic process lifecycle management

## Error Handling

MCP integration includes comprehensive error handling:

```python
from patterpunk.lib.mcp.exceptions import MCPConnectionError, MCPRequestError

try:
    chat = Chat().with_mcp_servers([server_config])
    response = chat.complete()
except MCPConnectionError as e:
    print(f"Failed to connect to MCP server {e.server_name}: {e}")
except MCPRequestError as e:
    print(f"MCP tool call failed on {e.server_name}: {e}")
```

## How It Works

1. **Server Discovery**: When `.with_mcp_servers()` is called, patterpunk connects to all MCP servers and discovers available tools
2. **Tool Integration**: MCP tools are converted to patterpunk's tool format and merged with existing tools
3. **Automatic Execution**: When the LLM calls an MCP tool, patterpunk automatically:
   - Identifies which MCP server provides the tool
   - Executes the tool call via JSON-RPC
   - Adds the result back to the conversation
   - Continues the completion if needed

## MCP Protocol Support

This implementation supports:
- ✅ MCP Protocol version 2025-03-26
- ✅ Streamable HTTP transport
- ✅ Stdio transport  
- ✅ Tool calling (`tools/list`, `tools/call`)
- ✅ Session management
- ✅ Error handling and timeouts
- 🔄 Resource access (planned)
- 🔄 Prompt templates (planned)

## Dependencies

- Core patterpunk: No additional dependencies
- HTTP transport: Requires `requests>=2.25.0` (installed with `patterpunk[mcp]`)
- Stdio transport: Uses Python standard library only