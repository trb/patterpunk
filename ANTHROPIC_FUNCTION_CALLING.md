# Anthropic Function Calling Implementation

This document explains how function calling (tool use) works with Anthropic Claude models in the Patterpunk library.

## Overview

Function calling allows Claude to call external functions/tools to gather information or perform actions. The implementation automatically converts OpenAI-style tool definitions to Anthropic's format and handles the response conversion back to the standard format.

## Key Features

- **Automatic Format Conversion**: OpenAI-style tool definitions are automatically converted to Anthropic's format
- **Unified Interface**: Same API as other providers - use the standard `Chat.with_tools()` method
- **Tool Call Handling**: Responses with tool calls are returned as `ToolCallMessage` objects
- **Error Handling**: Robust error handling for malformed tools and API errors

## How It Works

### 1. Tool Definition Format

Define tools using the standard OpenAI format:

```python
from patterpunk.llm.types import ToolDefinition

tools: ToolDefinition = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

### 2. Format Conversion

The `AnthropicModel._convert_tools_to_anthropic_format()` method automatically converts from:

**OpenAI Format:**
```json
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": { ... }
    }
}
```

**To Anthropic Format:**
```json
{
    "name": "get_weather",
    "description": "Get current weather",
    "input_schema": { ... }
}
```

### 3. Using Tools in Chat

```python
from patterpunk.llm.chat import Chat
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.messages import UserMessage

# Create chat with tools
chat = Chat(
    model=AnthropicModel(model="claude-3-5-sonnet-20240620")
).with_tools(tools)

# Add user message
chat = chat.add_message(UserMessage("What's the weather in Paris?"))

# Complete the chat
response = chat.complete()

# Check if Claude wants to call a tool
if chat.is_latest_message_tool_call:
    tool_calls = chat.latest_message.tool_calls
    # Execute the tools and provide results back to Claude
```

### 4. Tool Call Response Format

When Claude decides to use a tool, the response contains a `ToolCallMessage` with:

```python
{
    "id": "toolu_01234567890",
    "type": "function", 
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "Paris, France", "unit": "celsius"}'
    }
}
```

### 5. Executing Tool Calls

```python
import json

def execute_tool_call(tool_call):
    function_name = tool_call["function"]["name"]
    arguments = json.loads(tool_call["function"]["arguments"])
    
    # Your function registry
    if function_name == "get_weather":
        return get_weather(**arguments)
    
    return {"error": "Unknown function"}

# Execute all tool calls
if chat.is_latest_message_tool_call:
    for tool_call in chat.latest_message.tool_calls:
        result = execute_tool_call(tool_call)
        
        # Add result back to chat (implementation depends on your needs)
        chat = chat.add_message(UserMessage(f"Tool result: {json.dumps(result)}"))
    
    # Get final response
    final_response = chat.complete()
```

## Implementation Details

### AnthropicModel Changes

The `AnthropicModel` class has been enhanced with:

1. **Tool Conversion Method**: `_convert_tools_to_anthropic_format()`
2. **API Parameter Handling**: Tools are added to the API request when provided
3. **Response Processing**: Tool use responses are converted to `ToolCallMessage` objects

### Key Code Changes

```python
# In generate_assistant_message():
if tools:
    anthropic_tools = self._convert_tools_to_anthropic_format(tools)
    if anthropic_tools:
        api_params["tools"] = anthropic_tools

# Handle tool use response:
elif response.stop_reason == "tool_use":
    tool_calls = []
    for block in response.content:
        if block.type == "tool_use":
            tool_call = {
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input)
                }
            }
            tool_calls.append(tool_call)
    return ToolCallMessage(tool_calls)
```

## Best Practices

### 1. Detailed Tool Descriptions

Anthropic models perform better with detailed tool descriptions:

```python
{
    "name": "search_database",
    "description": "Search a product database for items matching specific criteria. This tool can filter by category, price range, and availability. It returns up to 10 results with product details including name, price, description, and stock status. Use this when users ask about finding or comparing products.",
    "parameters": { ... }
}
```

### 2. Error Handling

Always handle potential errors in tool execution:

```python
def execute_tool_call(tool_call):
    try:
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        
        if function_name not in FUNCTION_REGISTRY:
            return {"error": f"Unknown function: {function_name}"}
        
        return FUNCTION_REGISTRY[function_name](**arguments)
        
    except json.JSONDecodeError:
        return {"error": "Invalid JSON arguments"}
    except Exception as e:
        return {"error": f"Execution error: {str(e)}"}
```

### 3. Tool Choice Control

While not yet implemented, Anthropic supports tool choice control:

- `auto`: Let Claude decide whether to use tools (default)
- `any`: Force Claude to use one of the provided tools
- `tool`: Force Claude to use a specific tool
- `none`: Prevent Claude from using any tools

## Supported Models

Function calling works with all modern Claude models:

- `claude-3-5-sonnet-20240620`
- `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

## Example Usage

See `anthropic_function_calling_example.py` for a complete working example that demonstrates:

- Weather information retrieval
- Mathematical calculations
- Knowledge base searching
- Tool call execution and result handling

## Limitations

1. **Tool Results**: Currently, tool results must be added as regular messages. A dedicated `ToolResultMessage` class could be implemented for better structure.

2. **Parallel Tool Calls**: The current implementation handles multiple tool calls sequentially. Anthropic supports parallel execution.

3. **Tool Choice**: Advanced tool choice options (`any`, `tool`, `none`) are not yet exposed in the API.

4. **Streaming**: Tool use with streaming responses is not yet implemented.

## Future Enhancements

- Implement `ToolResultMessage` for structured tool results
- Add support for tool choice parameters
- Support for streaming tool use responses
- Better error handling and retry logic for tool execution
- Tool use analytics and logging