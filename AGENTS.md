# Agent Workflows

Agents are type-safe, reusable LLM workflows with automatic templating and structured output support. Use agents for multi-step processes, repeated operations, or when you need guaranteed type safety across workflow boundaries.

## Basic Agent

```python
from patterpunk.llm.agent import Agent
from patterpunk.llm.models.openai import OpenAiModel

class SummarizerAgent(Agent[str, str]):
    @property
    def model(self):
        return OpenAiModel(model="gpt-4")
    
    @property 
    def system_prompt(self):
        return "You are a concise summarizer. Respond with 1-2 sentences only."
    
    @property
    def _user_prompt_template(self):
        return "Summarize this text: {{ text }}"

agent = SummarizerAgent()
result = agent.execute("Long article text here...")
```

## Structured Output

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    topics: list[str]

class AnalyzerAgent(Agent[str, Analysis]):
    @property
    def model(self):
        return OpenAiModel(model="gpt-4")
    
    @property
    def system_prompt(self):
        return "Analyze text and return structured data."
    
    @property
    def _user_prompt_template(self):
        return "Analyze: {{ text }}"

# Returns typed Analysis instance automatically
analysis = AnalyzerAgent().execute("Sample text")
print(analysis.sentiment)  # Type-safe access
```

## Workflows

```python
from patterpunk.llm.chain import AgentChain, Parallel

# Sequential execution
chain = AgentChain([
    ExtractorAgent(),    # str -> str
    SummarizerAgent(),   # str -> str  
    AnalyzerAgent()      # str -> Analysis
])

result = chain.execute("Input text")

# Parallel execution
parallel = Parallel([
    SentimentAgent(),    # str -> str
    TopicAgent(),        # str -> str
    KeywordAgent()       # str -> str
])

# Mixed workflow
complex_chain = AgentChain([
    ExtractorAgent(),    # Extract key content
    parallel,            # Analyze in parallel -> list[str]
    # Final agent must handle list[str] input
])
```

## Practical Example

```python
# Document processing pipeline
class ExtractAgent(Agent[str, str]):
    @property
    def system_prompt(self):
        return "Extract the main content from this document, removing headers and footers."
    
    @property
    def _user_prompt_template(self):
        return "Document: {{ text }}"

class ClassifyAgent(Agent[str, str]):
    @property
    def system_prompt(self):
        return "Classify this content as: technical, business, or personal"
    
    @property 
    def _user_prompt_template(self):
        return "Content: {{ text }}"

# Type-safe pipeline
pipeline = AgentChain([ExtractAgent(), ClassifyAgent()])
classification = pipeline.execute(raw_document)
```

## Tool Integration

Override `prepare_chat()` to add tools to agent workflows:

```python
class ToolAgent(Agent[str, str]):
    @property
    def model(self):
        return OpenAiModel()

    @property
    def system_prompt(self):
        return "Use tools to help users."

    @property
    def _user_prompt_template(self):
        return "{{ text }}"

    def prepare_chat(self):
        return super().prepare_chat().with_tools([my_tool])

# Tools are auto-executed during agent.execute()
result = ToolAgent().execute("What's the weather in Tokyo?")
```

## Integration Notes

Agents work seamlessly with tools and MCP servers via the underlying Chat API. Override `execute()` method for custom logic when the default template→LLM→parse flow isn't sufficient.