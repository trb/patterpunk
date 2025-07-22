# Python Code Style Guide for Patterpunk

## Core Philosophy

Write Python code that prioritizes **functional programming principles** and **immutability** within the constraints of idiomatic Python style. Minimize object-oriented programming and inheritance while maintaining clean, maintainable code through focused, single-responsibility modules.

## Functional Programming Guidelines

### Immutability and Pure Functions

**PREFER**: Immutable data structures and pure functions that return new instances rather than modifying existing ones.

```python
# GOOD: Immutable chainable interface
def add_message(self, message: Message):
    new_chat = self.copy()
    new_chat.messages.append(message)
    return new_chat

# GOOD: Pure function with immutable operations
def process_messages(messages: List[Message]) -> List[ProcessedMessage]:
    return [process_single_message(msg) for msg in messages]
```

**AVOID**: In-place mutations and side effects when possible.

```python
# AVOID: Mutating existing state
def add_message(self, message: Message):
    self.messages.append(message)  # Mutates existing instance
    return self
```

### Functional Constructs

**PREFER**: List comprehensions, generator expressions, and built-in functional tools over explicit loops.

```python
# GOOD: List comprehension (idiomatic Python)
processed = [transform(item) for item in items if condition(item)]

# GOOD: Generator expression for memory efficiency
results = (expensive_operation(x) for x in large_dataset)

# ACCEPTABLE: Built-in functions
filtered_items = filter(predicate, items)
transformed = map(transform_func, items)
```

**AVOID**: Excessive lambda usage due to Python's limited lambda support.

```python
# AVOID: Complex lambdas
sorted_items = sorted(items, key=lambda x: x.attr1 if x.attr2 > 10 else x.attr3)

# PREFER: Named function for complex logic
def sort_key(item):
    return item.attr1 if item.attr2 > 10 else item.attr3

sorted_items = sorted(items, key=sort_key)
```

## Object-Oriented Programming Constraints

### Minimize Inheritance

**PRINCIPLE**: Use inheritance only when essential for the design. Most Python code should rely on composition, modules, and functions.

```python
# GOOD: Composition over inheritance
class Chat:
    def __init__(self, model: Model, messages: List[Message] = None):
        self.model = model
        self.messages = messages or []

# ACCEPTABLE: Inheritance for essential abstractions (like provider models)
class OpenAIModel(Model):
    def generate_assistant_message(self, messages: List[Message]) -> AssistantMessage:
        # Provider-specific implementation
        pass
```

**AVOID**: Deep inheritance hierarchies and unnecessary class-based solutions.

```python
# AVOID: Unnecessary inheritance
class BaseProcessor:
    def process(self): pass

class TextProcessor(BaseProcessor):
    def process(self): pass

class AdvancedTextProcessor(TextProcessor):
    def process(self): pass

# PREFER: Simple functions or composition
def process_text(text: str, options: ProcessingOptions) -> str:
    return apply_transformations(text, options.transformations)
```

### Prefer Functions and Modules

**PRINCIPLE**: Default to functions and modules. Use classes only when you need to maintain state or when implementing well-defined abstractions.

```python
# GOOD: Function-based approach
def extract_json(json_string: str) -> List[str]:
    # Implementation
    pass

def validate_message_format(message: Message) -> bool:
    # Implementation
    pass

# GOOD: Class when state management is essential
class Chat:
    def __init__(self, messages: List[Message], model: Model):
        self.messages = messages
        self.model = model
```

## File Organization Principles

### One Primary Export Per File

**PRINCIPLE**: Each file should have one primary class, function, or closely related set of utilities that can be imported from other modules.

```python
# GOOD: Single primary export
# file: patterpunk/llm/chat.py
class Chat:
    # Implementation

# GOOD: Related helper functions supporting the main export
# file: patterpunk/lib/extract_json.py
def extract_json(json_string: str) -> List[str]:
    # Main function

def _find_json_boundaries(text: str) -> List[Tuple[int, int]]:
    # Helper function used only by extract_json
```

### File Granularity Guidelines

**CREATE SEPARATE FILES** when:
- A class or function is imported from multiple other modules
- A component represents a distinct domain concept (e.g., `messages.py`, `chat.py`)
- A utility has broad applicability across the codebase

**KEEP IN SAME FILE** when:
- Helper functions are only used by one primary function in the same file
- Multiple small, related utilities form a cohesive collection (e.g., string manipulation helpers)
- Functions are tightly coupled and always used together

```python
# GOOD: Separate files for distinct concepts
# patterpunk/llm/chat.py - Chat class
# patterpunk/llm/messages.py - Message classes
# patterpunk/llm/models/openai.py - OpenAI provider

# GOOD: Related utilities in same file
# patterpunk/lib/text_utils.py
def normalize_whitespace(text: str) -> str: pass
def truncate_text(text: str, max_length: int) -> str: pass
def extract_sentences(text: str) -> List[str]: pass
```

### Module Structure

**PREFER**: Deep module nesting with focused responsibilities over flat structures with large files.

```python
# GOOD: Deep, focused structure
patterpunk/
├── llm/
│   ├── chat.py              # Chat class only
│   ├── messages.py          # Message classes
│   ├── agent.py             # Agent abstraction
│   └── models/
│       ├── base.py          # Abstract Model class
│       ├── openai.py        # OpenAI implementation
│       └── anthropic.py     # Anthropic implementation
└── lib/
    ├── extract_json.py      # JSON extraction utilities
    └── structured_output.py # Structured output handling
```

## Code Quality Standards

### Type Annotations

**REQUIRED**: Use comprehensive type annotations for all public interfaces.

```python
# GOOD: Complete type annotations
def process_messages(
    messages: List[Message], 
    model: Model,
    options: Optional[ProcessingOptions] = None
) -> List[AssistantMessage]:
    pass

# GOOD: Generic types for reusable components
from typing import TypeVar, Generic

T = TypeVar('T')
U = TypeVar('U')

class Agent(Generic[T, U]):
    def execute(self, input_data: T) -> U:
        pass
```

### Error Handling

**PREFER**: Explicit error types and functional error handling patterns.

```python
# GOOD: Specific exception types
class StructuredOutputParsingError(Exception):
    pass

class ProviderAuthenticationError(Exception):
    pass

# GOOD: Early returns for error conditions
def parse_structured_output(response: str, schema: Type[T]) -> T:
    if not response.strip():
        raise StructuredOutputParsingError("Empty response")
    
    try:
        return schema.parse_raw(response)
    except ValidationError as e:
        raise StructuredOutputParsingError(f"Failed to parse: {e}")
```

### Documentation

**MINIMAL COMMENTS**: Write self-documenting code. Use comments only for non-obvious business logic or complex algorithms.

```python
# GOOD: Self-documenting code
def extract_json_objects(text: str) -> List[str]:
    return [obj for obj in find_json_boundaries(text) if is_valid_json(obj)]

# GOOD: Comment for non-obvious logic
def calculate_token_usage(messages: List[Message]) -> int:
    # OpenAI counts system messages differently in token calculation
    base_tokens = sum(len(msg.content.split()) for msg in messages)
    system_message_penalty = len([msg for msg in messages if msg.role == "system"]) * 10
    return base_tokens + system_message_penalty
```

## Patterpunk-Specific Patterns

### Chainable Immutable Interface

**MAINTAIN**: The chainable, immutable pattern established in the Chat class.

```python
# GOOD: Chainable immutable methods
def add_message(self, message: Message) -> 'Chat':
    new_chat = self.copy()
    new_chat.messages.append(message)
    return new_chat

def with_model(self, model: Model) -> 'Chat':
    new_chat = self.copy()
    new_chat.model = model
    return new_chat
```

### Provider Abstraction

**ISOLATE**: Keep provider-specific code strictly within model files.

```python
# GOOD: Provider-agnostic interface in chat.py
def complete(self) -> 'Chat':
    response_message = self.model.generate_assistant_message(
        self.messages,
        structured_output=getattr(self.latest_message, "structured_output", None)
    )
    return self.add_message(response_message)

# GOOD: Provider-specific implementation in models/openai.py
def generate_assistant_message(self, messages: List[Message], **kwargs) -> AssistantMessage:
    # OpenAI-specific API calls and parameter mapping
    pass
```

### Structured Output Handling

**PATTERN**: Use Pydantic models for type-safe structured outputs with automatic fallback handling.

```python
# GOOD: Type-safe structured output
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    key_topics: List[str]

def analyze_text(text: str) -> AnalysisResult:
    response = chat.add_message(
        UserMessage(text, structured_output=AnalysisResult)
    ).complete()
    return response.latest_message.parsed_output
```

## Anti-Patterns to Avoid

1. **Large multi-purpose files** with unrelated classes and functions
2. **Deep inheritance hierarchies** for simple data transformations
3. **Stateful classes** when pure functions would suffice
4. **Provider-specific code** outside of model implementations
5. **Mutable shared state** that breaks the chainable interface pattern
6. **Complex lambda expressions** that reduce readability
7. **Unnecessary abstractions** that don't provide clear value

## Summary

Write Python code that embraces functional programming principles within Python's idioms, maintains strict separation of concerns through focused files and modules, and preserves the immutable, chainable interface patterns that make Patterpunk's API elegant and predictable.