import pytest

from patterpunk.llm.models.bedrock import BedrockModel
from patterpunk.llm.chat import Chat
from patterpunk.llm.messages import UserMessage


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "meta.llama3-8b-instruct-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
    ],
)
def test_simple_bedrock(model_id):
    bedrock = BedrockModel(model_id=model_id, temperature=0.1, top_p=0.98)

    print()
    print("Bedrock Models")
    print(BedrockModel.get_available_models())
    print()

    chat = Chat(model=bedrock)

    response = (
        chat.add_message(
            UserMessage(
                'What is the capital of Canada? Answer with JSON in this format: {"country": {"name": "the country the user asked for", "capital": "the capital of the country"}}. Think out loud and work step by step. Show your work. Do this before you generate the JSON response.'
            )
        )
        .complete()
        .latest_message.content
    )
