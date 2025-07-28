"""
Example usage of Patterpunk's prompt caching functionality.

This demonstrates how to use CacheChunk to optimize costs and performance
across different LLM providers (OpenAI, Anthropic, Bedrock, Google, Ollama).
"""

import sys
sys.path.insert(0, "/app")

from datetime import timedelta
from patterpunk.llm.types import CacheChunk
from patterpunk.llm.messages import SystemMessage, UserMessage


def basic_caching_example():
    """Basic example showing how to use cache chunks."""
    
    # Legacy approach (still works)
    system_msg_old = SystemMessage("You are a helpful assistant.")
    user_msg_old = UserMessage("What is Python?")
    
    # New approach with caching
    system_msg_new = SystemMessage([
        CacheChunk("You are a helpful assistant with expertise in:", cacheable=False),
        CacheChunk("Python programming, data science, web development...", cacheable=True)
    ])
    
    user_msg_new = UserMessage([
        CacheChunk("Large document context goes here...", cacheable=True, ttl=timedelta(hours=1)),
        CacheChunk("Question: What is Python?", cacheable=False)
    ])
    
    print("✓ Basic caching messages created successfully")


def document_analysis_example():
    """Example for document analysis with caching optimization."""
    
    large_document = """
    [Large document content - this would typically be thousands of tokens]
    This is a comprehensive document about machine learning algorithms...
    [Content continues for many paragraphs...]
    """
    
    analysis_instructions = """
    Please analyze documents using the following framework:
    1. Identify key themes and concepts
    2. Extract actionable insights
    3. Provide recommendations
    4. Highlight any risks or concerns
    """
    
    # Create a system message with cacheable instructions
    system_message = SystemMessage([
        CacheChunk("You are an expert analyst. ", cacheable=False),
        CacheChunk(analysis_instructions, cacheable=True, ttl=timedelta(hours=2))
    ])
    
    # Create user message with cacheable document content
    user_message = UserMessage([
        CacheChunk("Document to analyze:\n", cacheable=False),
        CacheChunk(large_document, cacheable=True, ttl=timedelta(hours=1)),
        CacheChunk("\nPlease provide a comprehensive analysis.", cacheable=False)
    ])
    
    print("✓ Document analysis messages created successfully")


def error_handling_and_fallback():
    """Demonstrate error handling with cache chunks."""
    
    try:
        # Even if caching fails, the content is still available as strings
        message = UserMessage([
            CacheChunk("Cacheable content", cacheable=True),
            CacheChunk("Regular content", cacheable=False)
        ])
        
        # This will always work, regardless of cache success/failure
        content_string = message.get_content_as_string()
        print(f"Full content: {content_string}")
        
        # Check if any content is cacheable
        if message.has_cacheable_content():
            print("This message has cacheable content")
        
        # Get individual chunks for provider-specific processing
        chunks = message.get_cache_chunks()
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}: cacheable={chunk.cacheable}, ttl={chunk.ttl}")
            
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    print("Running Patterpunk caching examples...")
    
    basic_caching_example()
    print("✓ Basic caching example completed")
    
    document_analysis_example()
    print("✓ Document analysis example completed")
    
    error_handling_and_fallback()
    print("✓ Error handling and fallback completed")
    
    print("\nAll examples completed successfully!")
    print("\nKey benefits of Patterpunk prompt caching:")
    print("- Up to 90% cost reduction on cached tokens")
    print("- Up to 85% latency improvement for long prompts")
    print("- Full backward compatibility with existing code")
    print("- Provider-specific optimizations")
    print("- Unified interface across OpenAI, Anthropic, Bedrock, Google, and Ollama")