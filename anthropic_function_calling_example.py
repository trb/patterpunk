#!/usr/bin/env python3
"""
Example demonstrating function calling with Anthropic Claude models.

This example shows how to:
1. Define tools/functions that Claude can call
2. Use the Chat interface with tools
3. Handle tool call responses
4. Execute the actual functions and return results

Note: You need to set PP_ANTHROPIC_API_KEY environment variable to run this example.
"""

import json
import math
from datetime import datetime
from typing import Dict, Any

from patterpunk.llm.chat import Chat
from patterpunk.llm.models.anthropic import AnthropicModel
from patterpunk.llm.messages import SystemMessage, UserMessage, AssistantMessage
from patterpunk.llm.types import ToolDefinition


def get_current_weather(location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
    """
    Mock function to get current weather.
    In a real application, this would call a weather API.
    """
    # Mock weather data
    weather_data = {
        "San Francisco, CA": {"temp": 72, "condition": "sunny", "humidity": 65},
        "New York, NY": {"temp": 68, "condition": "cloudy", "humidity": 70},
        "London, UK": {"temp": 60, "condition": "rainy", "humidity": 80},
        "Tokyo, Japan": {"temp": 75, "condition": "partly cloudy", "humidity": 60},
    }
    
    # Default to San Francisco if location not found
    data = weather_data.get(location, weather_data["San Francisco, CA"])
    
    # Convert to celsius if requested
    if unit.lower() == "celsius":
        data["temp"] = round((data["temp"] - 32) * 5/9, 1)
        unit_symbol = "°C"
    else:
        unit_symbol = "°F"
    
    return {
        "location": location,
        "temperature": f"{data['temp']}{unit_symbol}",
        "condition": data["condition"],
        "humidity": f"{data['humidity']}%",
        "timestamp": datetime.now().isoformat()
    }


def calculate_circle_area(radius: float) -> Dict[str, Any]:
    """
    Calculate the area of a circle given its radius.
    """
    if radius < 0:
        return {"error": "Radius cannot be negative"}
    
    area = math.pi * radius ** 2
    return {
        "radius": radius,
        "area": round(area, 4),
        "circumference": round(2 * math.pi * radius, 4),
        "diameter": radius * 2
    }


def search_knowledge_base(query: str, category: str = "general") -> Dict[str, Any]:
    """
    Mock function to search a knowledge base.
    In a real application, this would query a database or search engine.
    """
    # Mock knowledge base
    knowledge = {
        "python": [
            "Python is a high-level programming language known for its simplicity and readability.",
            "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            "Python has a rich ecosystem of libraries and frameworks for various applications."
        ],
        "ai": [
            "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
            "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.",
            "Deep learning uses neural networks with multiple layers to model complex patterns in data."
        ],
        "climate": [
            "Climate change refers to long-term shifts in global temperatures and weather patterns.",
            "The greenhouse effect is caused by gases that trap heat in Earth's atmosphere.",
            "Renewable energy sources like solar and wind power can help reduce carbon emissions."
        ]
    }
    
    # Simple keyword matching
    results = []
    query_lower = query.lower()
    
    for topic, facts in knowledge.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            results.extend(facts)
    
    # If no specific matches, return general results
    if not results and category == "general":
        results = knowledge.get("python", ["No specific information found."])
    
    return {
        "query": query,
        "category": category,
        "results": results[:3],  # Limit to top 3 results
        "total_found": len(results)
    }


# Define the tools in OpenAI format (will be converted to Anthropic format automatically)
TOOLS: ToolDefinition = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather information for a specific location. This function provides temperature, weather conditions, and humidity data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country, e.g. 'San Francisco, CA' or 'London, UK'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Defaults to fahrenheit."
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_circle_area",
            "description": "Calculate the area, circumference, and diameter of a circle given its radius. Useful for geometry calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "number",
                        "description": "The radius of the circle (must be positive)"
                    }
                },
                "required": ["radius"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search a knowledge base for information on various topics including Python, AI, climate change, and more. Returns relevant facts and information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or topic to look up"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["general", "technology", "science"],
                        "description": "The category to search within. Defaults to general."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Function registry for executing tool calls
FUNCTION_REGISTRY = {
    "get_current_weather": get_current_weather,
    "calculate_circle_area": calculate_circle_area,
    "search_knowledge_base": search_knowledge_base
}


def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """
    Execute a tool call and return the result as a string.
    
    :param tool_call: Tool call dictionary with id, type, and function info
    :return: JSON string with the function result
    """
    function_name = tool_call["function"]["name"]
    arguments_str = tool_call["function"]["arguments"]
    
    try:
        # Parse the arguments JSON string
        arguments = json.loads(arguments_str)
        
        # Get the function from registry
        if function_name not in FUNCTION_REGISTRY:
            return json.dumps({"error": f"Unknown function: {function_name}"})
        
        func = FUNCTION_REGISTRY[function_name]
        
        # Execute the function
        result = func(**arguments)
        
        return json.dumps(result, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON arguments: {e}"})
    except TypeError as e:
        return json.dumps({"error": f"Invalid function arguments: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Function execution error: {e}"})


def demonstrate_function_calling():
    """Demonstrate function calling with various examples."""
    
    print("🤖 Anthropic Function Calling Demo")
    print("=" * 50)
    
    # Initialize the chat with Anthropic model and tools
    chat = Chat(
        model=AnthropicModel(
            model="claude-3-5-sonnet-20240620",
            temperature=0.1,
            max_tokens=2000
        )
    ).with_tools(TOOLS)
    
    # Add system message
    chat = chat.add_message(SystemMessage(
        """You are a helpful assistant with access to several tools. You can:
        1. Get weather information for any location
        2. Calculate circle properties (area, circumference, diameter)
        3. Search a knowledge base for information on various topics
        
        When a user asks for information that requires using these tools, call the appropriate function(s) to get the data, then provide a comprehensive response based on the results."""
    ))
    
    # Example 1: Weather query
    print("\n📍 Example 1: Weather Query")
    print("-" * 30)
    
    chat = chat.add_message(UserMessage(
        "What's the weather like in San Francisco and London? Please compare them."
    ))
    
    response = chat.complete()
    print(f"User: What's the weather like in San Francisco and London? Please compare them.")
    
    # Handle tool calls if any
    if chat.is_latest_message_tool_call:
        print(f"Claude: [Calling tools...]")
        
        # Execute each tool call
        for tool_call in chat.latest_message.tool_calls:
            function_name = tool_call["function"]["name"]
            print(f"  → Calling {function_name}")
            
            # Execute the function
            result = execute_tool_call(tool_call)
            
            # Add the result back to the chat
            # Note: In a full implementation, you'd use a ToolResultMessage
            # For now, we'll add it as a user message
            chat = chat.add_message(UserMessage(
                f"Tool result for {function_name}: {result}"
            ))
        
        # Get the final response
        response = chat.complete()
    
    print(f"Claude: {response.latest_message.content}")
    
    # Example 2: Math calculation
    print("\n🔢 Example 2: Circle Calculation")
    print("-" * 30)
    
    chat = chat.add_message(UserMessage(
        "I have a circular garden with a radius of 5 meters. Can you calculate its area and tell me how much fencing I'd need for the perimeter?"
    ))
    
    response = chat.complete()
    print(f"User: I have a circular garden with a radius of 5 meters. Can you calculate its area and tell me how much fencing I'd need for the perimeter?")
    
    if chat.is_latest_message_tool_call:
        print(f"Claude: [Calling tools...]")
        
        for tool_call in chat.latest_message.tool_calls:
            function_name = tool_call["function"]["name"]
            print(f"  → Calling {function_name}")
            
            result = execute_tool_call(tool_call)
            chat = chat.add_message(UserMessage(
                f"Tool result for {function_name}: {result}"
            ))
        
        response = chat.complete()
    
    print(f"Claude: {response.latest_message.content}")
    
    # Example 3: Knowledge search
    print("\n🔍 Example 3: Knowledge Search")
    print("-" * 30)
    
    chat = chat.add_message(UserMessage(
        "Can you tell me about machine learning and how it relates to AI?"
    ))
    
    response = chat.complete()
    print(f"User: Can you tell me about machine learning and how it relates to AI?")
    
    if chat.is_latest_message_tool_call:
        print(f"Claude: [Calling tools...]")
        
        for tool_call in chat.latest_message.tool_calls:
            function_name = tool_call["function"]["name"]
            print(f"  → Calling {function_name}")
            
            result = execute_tool_call(tool_call)
            chat = chat.add_message(UserMessage(
                f"Tool result for {function_name}: {result}"
            ))
        
        response = chat.complete()
    
    print(f"Claude: {response.latest_message.content}")


if __name__ == "__main__":
    try:
        demonstrate_function_calling()
        print("\n✅ Function calling demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
        # Check if it's an API key issue
        if "api" in str(e).lower() or "auth" in str(e).lower():
            print("\n💡 Make sure you have set the PP_ANTHROPIC_API_KEY environment variable.")