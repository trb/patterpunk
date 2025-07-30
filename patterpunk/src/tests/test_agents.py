import logging
from dataclasses import dataclass
from typing import List
from pydantic import BaseModel

from patterpunk.llm.agent import Agent
from patterpunk.llm.chain import AgentChain, Parallel
from patterpunk.llm.models.openai import OpenAiModel
from patterpunk.logger import logger, logger_llm


@dataclass
class SimpleInput:
    text: str


class SummaryOutput(BaseModel):
    summary: str
    word_count: int
    sentiment: str


class StringOutputAgent(Agent[SimpleInput, str]):
    def __init__(self):
        self._model = OpenAiModel(model="gpt-3.5-turbo", temperature=0.1)

    @property
    def model(self):
        return self._model

    @property
    def system_prompt(self) -> str:
        return "You are a helpful assistant. Respond concisely in one sentence."

    @property
    def _user_prompt_template(self) -> str:
        return "Please summarize this text: {{ text }}"


class StructuredOutputAgent(Agent[SimpleInput, SummaryOutput]):
    def __init__(self):
        self._model = OpenAiModel(model="gpt-3.5-turbo", temperature=0.1)

    @property
    def model(self):
        return self._model

    @property
    def system_prompt(self) -> str:
        return "You are a text analyzer. Analyze the provided text and return structured output."

    @property
    def _user_prompt_template(self) -> str:
        return "Analyze this text: {{ text }}"


class CustomExecuteAgent(Agent[SimpleInput, str]):
    def __init__(self):
        self._model = OpenAiModel(model="gpt-3.5-turbo", temperature=0.1)

    @property
    def model(self):
        return self._model

    @property
    def system_prompt(self) -> str:
        return "You are a helpful assistant."

    @property
    def _user_prompt_template(self) -> str:
        return "Process: {{ text }}"

    def execute(self, input_data: SimpleInput) -> str:
        return f"CUSTOM_EXECUTE: {input_data.text.upper()}"


class SimpleUppercaseAgent(Agent[str, str]):
    def __init__(self):
        self._model = OpenAiModel(model="gpt-3.5-turbo", temperature=0.0)

    @property
    def model(self):
        return self._model

    @property
    def system_prompt(self) -> str:
        return "Convert the input text to uppercase. Respond with ONLY the uppercase version, no other text."

    @property
    def _user_prompt_template(self) -> str:
        return "{{ text }}"


class SimpleLowercaseAgent(Agent[str, str]):
    def __init__(self):
        self._model = OpenAiModel(model="gpt-3.5-turbo", temperature=0.0)

    @property
    def model(self):
        return self._model

    @property
    def system_prompt(self) -> str:
        return "Convert the input text to lowercase. Respond with ONLY the lowercase version, no other text."

    @property
    def _user_prompt_template(self) -> str:
        return "{{ text }}"


class SimplePrefixAgent(Agent[str, str]):
    def __init__(self, prefix: str):
        self._model = OpenAiModel(model="gpt-3.5-turbo", temperature=0.0)
        self.prefix = prefix

    @property
    def model(self):
        return self._model

    @property
    def system_prompt(self) -> str:
        return f"Add the prefix '{self.prefix}' to the beginning of the input text. Respond with ONLY the prefixed text, no other words."

    @property
    def _user_prompt_template(self) -> str:
        return "{{ text }}"


def test_agent_string_output():
    logger.setLevel(logging.ERROR)
    logger_llm.setLevel(logging.ERROR)

    agent = StringOutputAgent()
    input_data = SimpleInput(
        text="Machine learning is a fascinating field of artificial intelligence."
    )

    result = agent.execute(input_data)

    assert isinstance(result, str)
    assert len(result) > 0
    print(f"String output result: {result}")


def test_agent_structured_output():
    logger.setLevel(logging.ERROR)
    logger_llm.setLevel(logging.ERROR)

    agent = StructuredOutputAgent()
    input_data = SimpleInput(
        text="Artificial intelligence and machine learning are revolutionizing technology. They enable computers to learn and make decisions."
    )

    result = agent.execute(input_data)

    assert isinstance(result, SummaryOutput)
    assert isinstance(result.summary, str)
    assert isinstance(result.word_count, int)
    assert isinstance(result.sentiment, str)
    assert len(result.summary) > 0
    assert result.word_count > 0
    assert result.sentiment.lower() in ["positive", "negative", "neutral"]
    print(f"Structured output result: {result}")


def test_agent_custom_execute():
    logger.setLevel(logging.ERROR)
    logger_llm.setLevel(logging.ERROR)

    agent = CustomExecuteAgent()
    input_data = SimpleInput(text="hello world")

    result = agent.execute(input_data)

    assert result == "CUSTOM_EXECUTE: HELLO WORLD"
    print(f"Custom execute result: {result}")


def test_agent_chain_sequential():
    logger.setLevel(logging.ERROR)
    logger_llm.setLevel(logging.ERROR)

    uppercase_agent = SimpleUppercaseAgent()
    prefix_agent = SimplePrefixAgent("PREFIX:")

    chain = AgentChain([uppercase_agent, prefix_agent])

    result = chain.execute("hello world")

    assert isinstance(result, str)
    assert "PREFIX:" in result
    assert "HELLO" in result.upper()
    print(f"Sequential chain result: {result}")


def test_agent_chain_with_parallel():
    logger.setLevel(logging.ERROR)
    logger_llm.setLevel(logging.ERROR)

    uppercase_agent = SimpleUppercaseAgent()
    lowercase_agent = SimpleLowercaseAgent()

    parallel_step = Parallel([uppercase_agent, lowercase_agent])
    chain = AgentChain([parallel_step])

    result = chain.execute("Hello World")

    assert isinstance(result, list)
    assert len(result) == 2

    uppercase_result = result[0]
    lowercase_result = result[1]

    assert isinstance(uppercase_result, str)
    assert isinstance(lowercase_result, str)

    has_uppercase = any("HELLO" in r.upper() for r in result)
    has_lowercase = any("hello" in r.lower() for r in result)
    assert has_uppercase or has_lowercase

    print(f"Parallel execution results: {result}")


def test_complex_agent_chain():
    logger.setLevel(logging.ERROR)
    logger_llm.setLevel(logging.ERROR)

    uppercase_agent = SimpleUppercaseAgent()
    prefix_agent_1 = SimplePrefixAgent("STEP1:")
    prefix_agent_2 = SimplePrefixAgent("STEP2:")

    parallel_step = Parallel([prefix_agent_1, prefix_agent_2])

    chain = AgentChain([uppercase_agent, parallel_step])

    result = chain.execute("test input")

    assert isinstance(result, list)
    assert len(result) == 2

    for item in result:
        assert isinstance(item, str)
        assert "STEP" in item

    print(f"Complex chain result: {result}")


def test_agent_type_validation():
    logger.setLevel(logging.ERROR)
    logger_llm.setLevel(logging.ERROR)

    string_agent = StringOutputAgent()
    structured_agent = StructuredOutputAgent()

    string_input = SimpleInput(text="test text")

    string_result = string_agent.execute(string_input)
    structured_result = structured_agent.execute(string_input)

    assert type(string_result) == str
    assert type(structured_result) == SummaryOutput

    assert hasattr(structured_result, "summary")
    assert hasattr(structured_result, "word_count")
    assert hasattr(structured_result, "sentiment")

    print(
        f"Type validation passed - String: {type(string_result)}, Structured: {type(structured_result)}"
    )
