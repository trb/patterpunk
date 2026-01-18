# Tool Calling

Patterpunk provides automatic function-to-tool conversion and MCP (Model Context Protocol) integration. By default, all tools (both function tools and MCP tools) execute automatically when the model calls them.

To handle tool calls manually, pass `execute_tools=False` to `complete()`:

```python
response = chat.complete(execute_tools=False)
if response.is_latest_message_tool_call:
    # Manual handling here
```

## Function Tools

### Automatic Conversion

```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.models.openai import OpenAiModel

def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get current weather for a location.
    
    Args:
        location: City name or coordinates
        unit: Temperature unit (fahrenheit, celsius)
    """
    return f"Weather in {location}: 72°{unit[0].upper()}"

def search_web(query: str, max_results: int = 5) -> list[str]:
    """Search the web for information."""
    return [f"Result {i}: {query}" for i in range(max_results)]

chat = Chat(model=OpenAiModel()).with_tools([get_weather, search_web])
response = chat.add_message("What's the weather in Paris?").complete()

# Tool calls are auto-executed! Response contains the final answer.
print(response.latest_message.content)  # "The weather in Paris is 72°F"
```

### Advanced Function Features

```python
from typing import Optional, Union, List, Dict

def process_data(
    items: List[Dict[str, Union[str, int]]],
    config: Optional[Dict[str, bool]] = None,
    threshold: float = 0.5
) -> bool:
    """Process complex nested data structures.
    
    Args:
        items: List of data items with mixed types
        config: Optional processing configuration
        threshold: Processing threshold value
    """
    return len(items) > threshold

# Automatic schema generation handles:
# - Complex nested types (List[Dict[str, Union[str, int]]])
# - Optional parameters with defaults
# - Google-style docstring parsing for descriptions
# - Type validation and conversion
```

### Manual Tool Definitions

```python
# For cases requiring precise control over tool schema
manual_tools = [{
    "type": "function",
    "function": {
        "name": "custom_tool",
        "description": "Precisely controlled tool definition",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "enum": ["a", "b", "c"]}
            },
            "required": ["param"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

chat = chat.with_tools(manual_tools)
```

## MCP Integration

### Server Configuration

```python
from patterpunk.lib.mcp import MCPServerConfig

# HTTP transport (remote/containerized servers)
weather_server = MCPServerConfig(
    name="weather-server",
    url="http://mcp-weather-server:8000/mcp",
    timeout=30.0
)

# Stdio transport (local subprocess servers)
filesystem_server = MCPServerConfig(
    name="filesystem",
    command=["python", "-m", "mcp_filesystem"],
    env={"MCP_FILESYSTEM_ROOT": "/workspace"}
)

chat = chat.with_mcp_servers([weather_server, filesystem_server])
```

### Automatic Execution

```python
# MCP tools execute automatically - no manual handling needed
response = chat.add_message("List files in /workspace").complete()

# Tool calls are executed and results are automatically added to conversation
print(response.latest_message.content)  # Contains results from filesystem tool
```

### Mixed Tool Usage

```python
# Combine function tools and MCP servers - all execute automatically
def local_function(data: str) -> str:
    """Local function tool."""
    return data.upper()

chat = (Chat(model=OpenAiModel())
        .with_tools([local_function])
        .with_mcp_servers([filesystem_server]))

response = chat.add_message("Process 'hello' and list files").complete()

# Both function tools and MCP tools are auto-executed
print(response.latest_message.content)  # Contains results from both tools
```

## ToolCallMessage Structure

```python
# Access tool call details
tool_message = response.latest_message  # ToolCallMessage instance

for tool_call in tool_message.tool_calls:
    call_id = tool_call["id"]                    # Unique call identifier
    function_name = tool_call["function"]["name"] # Function to execute
    arguments_json = tool_call["function"]["arguments"] # JSON string
    
    # Parse arguments safely
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        arguments = {}
```

## Integration Patterns

### Agent Integration

```python
from patterpunk.llm.agent import Agent

class ToolAgent(Agent[str, str]):
    @property
    def model(self):
        return OpenAiModel()
    
    @property
    def system_prompt(self):
        return "Use available tools to help users."
    
    @property
    def _user_prompt_template(self):
        return "{{ text }}"
    
    def prepare_chat(self):
        return super().prepare_chat().with_tools([get_weather])

# Tools are available within agent execution
agent = ToolAgent()
result = agent.execute("What's the weather in Tokyo?")
```

### Error Handling

With auto-execution (default), tool errors are automatically sent to the model as error results. The model can then respond appropriately (apologize, ask for clarification, try again).

For manual execution with custom error handling:

```python
from patterpunk.llm.messages import ToolResultMessage

response = chat.complete(execute_tools=False)

if response.is_latest_message_tool_call:
    for tool_call in response.latest_message.tool_calls:
        call_id = tool_call["id"]
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        try:
            result = execute_function(**arguments)
            chat = chat.add_message(ToolResultMessage(
                content=str(result),
                call_id=call_id,
                function_name=function_name
            ))
        except Exception as e:
            # Report error to the model with is_error flag
            chat = chat.add_message(ToolResultMessage(
                content=f"Tool {function_name} failed: {e}",
                call_id=call_id,
                function_name=function_name,
                is_error=True
            ))

    response = chat.complete()
```

## Controlling Tool Execution

By default, `complete()` executes all tool calls automatically. To handle them manually:

```python
# Disable auto-execution
response = chat.complete(execute_tools=False)

if response.is_latest_message_tool_call:
    for tool_call in response.latest_message.tool_calls:
        call_id = tool_call["id"]
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        # Custom execution logic here
        result = my_custom_executor(function_name, arguments)

        chat = chat.add_message(ToolResultMessage(
            content=str(result),
            call_id=call_id,
            function_name=function_name
        ))

    # Continue conversation with results
    response = chat.complete()
```

Use manual execution when you need:
- Custom execution environments or sandboxing
- Pre-execution validation or approval flows
- Logging or monitoring of tool calls
- Conditional execution based on tool type

## Key Differences

**Function Tools:**
- Automatic schema generation from Python functions
- Automatic execution by default (pass `execute_tools=False` for manual control)
- Immediate availability (no external dependencies)
- Best for simple, fast operations

**MCP Tools:**
- External server communication via HTTP/stdio
- Automatic execution by default (same as function tools)
- Requires optional dependencies (requests for HTTP)
- Best for complex operations, external services, or sandboxed execution