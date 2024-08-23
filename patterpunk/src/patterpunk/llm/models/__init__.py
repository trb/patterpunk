import patterpunk.config


def get_available_models():
    models = []

    if patterpunk.config.anthropic:
        from .anthropic import AnthropicModel

        models.append(AnthropicModel)

    if patterpunk.config.openai:
        from .openai import OpenAiModel

        models.append(OpenAiModel)

    if patterpunk.config.ollama:
        from .ollama import OllamaModel

        models.append(OllamaModel)
