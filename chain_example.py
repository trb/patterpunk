from dataclasses import dataclass
from typing import List
from patterpunk import Agent, AgentChain, Parallel
from patterpunk.llm.models.openai import OpenAIModel


@dataclass
class CodePrompt:
    prompt: str


@dataclass
class CodeSnippet:
    code: str


class GenerateCodeAgent(Agent[CodePrompt, CodeSnippet]):
    def __init__(self, model_name: str):
        self._model_name = model_name

    @property
    def model(self):
        return OpenAIModel(model=self._model_name)

    @property
    def system_prompt(self) -> str:
        return "You are an expert at writing Python code. You will be given a prompt and you must return a single block of Python code."

    @property
    def _user_prompt_template(self) -> str:
        return "{{ prompt }}"


@dataclass
class CodeSnippets:
    snippets: List[CodeSnippet]


@dataclass
class BestCodeSnippet:
    best_snippet: CodeSnippet


class QuorumAgent(Agent[CodeSnippets, BestCodeSnippet]):
    @property
    def model(self):
        return OpenAIModel(model="gpt-4")

    @property
    def system_prompt(self) -> str:
        return "You are an expert at reviewing Python code. You will be given a list of code snippets and you must pick the best one."

    @property
    def _user_prompt_template(self) -> str:
        return """
        {% for snippet in snippets %}
        Snippet {{ loop.index }}:
        ```python
        {{ snippet.code }}
        ```
        {% endfor %}
        """


# You need to have an OPENAI_API_KEY environment variable set to run this example
if __name__ == "__main__":
    chain = AgentChain(
        steps=[
            Parallel(
                agents=[
                    GenerateCodeAgent(model_name="gpt-3.5-turbo"),
                    GenerateCodeAgent(model_name="gpt-4"),
                    GenerateCodeAgent(model_name="gpt-4-turbo"),
                ]
            ),
            QuorumAgent(),
        ]
    )

    initial_prompt = CodePrompt(prompt="A function that calculates the nth fibonacci number.")
    best_snippet = chain.execute(initial_prompt)
    print(best_snippet)
