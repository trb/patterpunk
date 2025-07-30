"""
Example usage of Patterpunk's prompt caching functionality.

This demonstrates how to use CacheChunk to optimize costs and performance
across different LLM providers (OpenAI, Anthropic, Bedrock, Google, Ollama).
"""

from datetime import timedelta
from patterpunk.llm.types import CacheChunk, TextChunk
from patterpunk.llm.messages import SystemMessage, UserMessage
from patterpunk.llm.chat import Chat


def basic_caching_example():
    """Basic example showing how to use cache chunks."""
    
    # Legacy approach (still works)
    system_msg_old = SystemMessage("You are a helpful assistant.")
    user_msg_old = UserMessage("What is Python?")
    
    # New approach with caching and TextChunk
    system_msg_new = SystemMessage([
        TextChunk("You are a helpful assistant with expertise in:"),
        CacheChunk("Python programming, data science, web development...", cacheable=True)
    ])
    
    user_msg_new = UserMessage([
        CacheChunk("Large document context goes here...", cacheable=True, ttl=timedelta(hours=1)),
        TextChunk("Question: What is Python?")
    ])
    
    # Both approaches work the same way
    chat = Chat().add_message(system_msg_new).add_message(user_msg_new)
    # response = chat.complete()


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
        TextChunk("You are an expert analyst. "),
        CacheChunk(analysis_instructions, cacheable=True, ttl=timedelta(hours=2))
    ])
    
    # Create user message with cacheable document content
    user_message = UserMessage([
        TextChunk("Document to analyze:\n"),
        CacheChunk(large_document, cacheable=True, ttl=timedelta(hours=1)),
        TextChunk("\nPlease provide a comprehensive analysis.")
    ])
    
    chat = Chat().add_message(system_message).add_message(user_message)
    # response = chat.complete()


def multi_query_example():
    """Example showing how to reuse cached content for multiple queries."""
    
    codebase_context = """
    [Large codebase context - repository structure, key files, etc.]
    This codebase is a web application built with FastAPI...
    [Content continues...]
    """
    
    # Cache the codebase context for reuse across multiple queries
    base_system = SystemMessage([
        TextChunk("You are a senior software engineer reviewing this codebase:\n"),
        CacheChunk(codebase_context, cacheable=True, ttl=timedelta(hours=2))
    ])
    
    # Multiple queries reusing the same cached context
    queries = [
        "What are the main security vulnerabilities?",
        "How can we improve performance?",
        "What tests are missing?",
        "How is error handling implemented?"
    ]
    
    chat = Chat().add_message(base_system)
    
    for query in queries:
        user_message = UserMessage([
            TextChunk("Question: "),
            TextChunk(query)
        ])
        
        query_chat = chat.add_message(user_message)
        # response = query_chat.complete()
        # Process response for each query


def provider_specific_optimization():
    """Examples optimized for specific providers."""
    
    # OpenAI - Optimize for prefix caching
    openai_optimized = UserMessage([
        CacheChunk("Static context at the beginning for prefix caching", cacheable=True),
        CacheChunk("Dynamic query content", cacheable=False)
    ])
    
    # Anthropic - Use explicit cache controls
    anthropic_optimized = UserMessage([
        CacheChunk("Context section 1", cacheable=True, ttl=timedelta(minutes=5)),
        CacheChunk("Context section 2", cacheable=True, ttl=timedelta(hours=1)),  # Extended TTL
        CacheChunk("Current query", cacheable=False)
    ])
    
    # Bedrock - Use cache points
    bedrock_optimized = UserMessage([
        CacheChunk("Document content", cacheable=True),
        CacheChunk("Analysis request", cacheable=False)
    ])
    
    # Google - Ensure content meets 32k token minimum
    large_content = "A" * 50000  # Simulate large content for Google's minimum
    google_optimized = UserMessage([
        CacheChunk(large_content, cacheable=True, ttl=timedelta(hours=1)),
        CacheChunk("Query about the content", cacheable=False)
    ])
    
    # Ollama - Gracefully degrades to string content (no caching)
    ollama_message = UserMessage([
        CacheChunk("This will be concatenated", cacheable=True),
        CacheChunk(" into a single string", cacheable=False)
    ])


def cost_optimization_patterns():
    """Patterns for maximizing cost savings."""
    
    # Pattern 1: Cache expensive-to-compute content
    expensive_computation = """
    [Results of expensive computation, analysis, or processing]
    This represents content that was expensive to generate...
    """
    
    message1 = UserMessage([
        CacheChunk(expensive_computation, cacheable=True, ttl=timedelta(hours=4)),
        CacheChunk("New question about the results", cacheable=False)
    ])
    
    # Pattern 2: Hierarchical caching for different content types
    message2 = SystemMessage([
        CacheChunk("Base system instructions", cacheable=True, ttl=timedelta(days=1)),
        CacheChunk("Domain-specific context", cacheable=True, ttl=timedelta(hours=8)),
        CacheChunk("Session-specific settings", cacheable=False)
    ])
    
    # Pattern 3: Template-based caching
    def create_analysis_message(document, query):
        return UserMessage([
            CacheChunk("Analysis Template:\n1. Read document\n2. Extract key points\n3. Answer question\n\n", 
                      cacheable=True, ttl=timedelta(hours=12)),
            CacheChunk(f"Document: {document}\n", cacheable=True, ttl=timedelta(hours=2)),
            CacheChunk(f"Question: {query}", cacheable=False)
        ])


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
        # Application continues to work normally


if __name__ == "__main__":
    print("Running Patterpunk caching examples...")
    
    basic_caching_example()
    print("✓ Basic caching example completed")
    
    document_analysis_example()
    print("✓ Document analysis example completed")
    
    multi_query_example()
    print("✓ Multi-query example completed")
    
    provider_specific_optimization()
    print("✓ Provider-specific optimization examples completed")
    
    cost_optimization_patterns()
    print("✓ Cost optimization patterns completed")
    
    error_handling_and_fallback()
    print("✓ Error handling and fallback completed")
    
    print("\nAll examples completed successfully!")
    print("\nKey benefits of Patterpunk prompt caching:")
    print("- Up to 90% cost reduction on cached tokens")
    print("- Up to 85% latency improvement for long prompts")
    print("- Full backward compatibility with existing code")
    print("- Provider-specific optimizations")
    print("- Unified interface across OpenAI, Anthropic, Bedrock, Google, and Ollama")