from patterpunk.config import DEFAULT_MODEL
from patterpunk.llm.openai import OpenAiModel


def default_model():
    return OpenAiModel(model=DEFAULT_MODEL)
