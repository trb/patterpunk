from dataclasses import dataclass
from typing import List
from patterpunk import Agent
from patterpunk.llm.models.openai import OpenAIModel


@dataclass
class SearchResult:
    url: str
    id: str
    preview: str


@dataclass
class SearchResults:
    search_results: List[SearchResult]
    request: str


class PickBestSearchResult(Agent[SearchResults, SearchResult]):
    @property
    def model(self):
        return OpenAIModel(model="gpt-4")

    @property
    def system_prompt(self) -> str:
        return "You are an expert at picking the best search result for a given request. You will be given a list of search results and a request. You must pick the best search result that satisfies the request."

    @property
    def _user_prompt_template(self) -> str:
        return """
        Request: {{ request }}

        Search Results:
        {% for result in search_results %}
        - {{ result.url }} (ID: {{ result.id }}): {{ result.preview }}
        {% endfor %}
        """


# You need to have an OPENAI_API_KEY environment variable set to run this example
if __name__ == "__main__":
    agent = PickBestSearchResult()

    search_results = SearchResults(
        search_results=[
            SearchResult(
                url="https://www.example.com",
                id="1",
                preview="This is an example search result.",
            ),
            SearchResult(
                url="https://www.anotherexample.com",
                id="2",
                preview="This is another example search result.",
            ),
        ],
        request="Which search result is better?",
    )

    best_result = agent.execute(search_results)

    print(best_result)
