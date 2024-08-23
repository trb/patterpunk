import inspect
from typing import Callable, List

from patterpunk.llm.chat import Chat
from patterpunk.llm.defaults import default_model
from patterpunk.llm.messages import SystemMessage
from patterpunk.llm.models.openai import Model, OpenAiModel
from patterpunk.logger import logger
from pydantic import BaseModel, ValidationError

"""
@todo a bunch of things since we only have a skeleton so far:

- get params and inject into prompts
    - be smart about this by converting objects to str or repr first, allowing user to inject complex types
    - create model for openai
    - if to-wrap function had return types, generate a json model for it
        - if it's not a pydantic type make one up for the return value like {"result": type}
        - add system message to messages instructing llm to generate json
    - send message to openai
    - if the to-wrap function has a body, call it. make sure it returns a generator function, if not, throw an error
        - call the generator function with the messages from llm so far (including assistantmessage)
        - do that until generator is done
        - set the additional messages as return messages
    - if return type of to-wrap function is not str, check that assistant messages contain json
        - try to deserialize into the return type for each json found
        - deserializer will need stack-tracking of {} or []
            - for each char in message, check if [ or {
            - if it is, put on stack. for each new [ or { (depending on which one was found first) put on stack
            - if ] or } encountered, pop from stack (again, which one depends on first one)
            - when stack empty -> that's the json
            - if stack not empty but message over -> exception
        - if not deserialize -> exception
        - if multiple deserialize, return in order
"""


class BadJsonResponseError(Exception):
    pass


class UnexpectedFunctionCallError(Exception):
    pass


def get_json_system_message(model: BaseModel):
    return SystemMessage(
        f"""
                You are given a json schema and your task is to generate json that conforms to the schema.
                Take the data from the previous messages and generate a json object that follows the specified schema.
                Pay close attention to the schema. For example, if an attribute is an array, try to find all the
                data in the previous prompts that match the attribute and return all of them.

                Only extract data. Do not fill in the blanks, summarize, shorten or create new output.

                Generate JSON that is valid for this json schema:
                {model.model_json_schema()}"""
    )


def extract_json_gpt_35(return_type: BaseModel, chat: Chat):
    gpt35_system_message = get_json_system_message(return_type).set_model(
        OpenAiModel(model="gpt-3.5-turbo-16k", temperature=0.01, top_p=0.2)
    )
    json_chat = chat.add_message(gpt35_system_message)
    json_chat = json_chat.complete()
    try:
        data = return_type.model_validate_json(json_chat.messages[-1].content)
        logger.info("Parsed returned json successfully with gpt-3.5")
        return data
    except ValidationError as error:
        logger.debug(
            f"Failed to parse json message with gpt-3.5, message: {json_chat.messages[-1].content}",
            exc_info=error,
        )
        return None


def extract_json_gpt_4(return_type: BaseModel, chat: Chat):
    gpt4_system_message = get_json_system_message(return_type).set_model(
        OpenAiModel(model="gpt-4", temperature=0.01, top_p=0.2)
    )
    json_chat = chat.add_message(gpt4_system_message)
    json_chat = json_chat.complete()
    try:
        print(json_chat.messages[-1].content)
        data = return_type.model_validate_json(json_chat.messages[-1].content)
        logger.info("Parsed returned json successfully with gpt-4")
        return data
    except ValidationError as error:
        logger.warning(
            f"Failed to parse json message with gpt-4, message: {json_chat.messages[-1].content}",
            exc_info=error,
        )
        return None


def chatcomplete(
    *original_messages,
    model: Model = None,
    infix_processor=None,
    functions: List[Callable] = None,
):
    """
    Turns a python function into an LLM call.

    :param original_messages: List of messages to send to the LLM
    :param model: Optional: Model to use for the LLM, will use default model if not specified
    :param infix_processor: Optional: Function that receives a chat object and returns chat object. Runs after original messages are processed by LLM but before JSON conversion is attempted
    :return:
    """

    def create_chat_completion(chat_completion_request):
        signature = inspect.signature(chat_completion_request)

        llm_model = default_model() if model is None else model

        def chat_complete(*args, **kwargs):
            parameters = {}
            all_parameters = list(signature.parameters.values())
            for index, arg in enumerate(args):
                parameter = all_parameters[index]
                parameters[parameter.name] = arg

            parameters.update(kwargs)
            if hasattr(signature.return_annotation, "model_json_schema"):
                parameters["return_type"] = (
                    signature.return_annotation.model_json_schema()
                )

            messages = [message.prepare(parameters) for message in original_messages]

            chat = Chat(messages=messages, model=llm_model, functions=functions)
            chat = chat.complete()

            if infix_processor:
                logger.debug("Running infix processor")
                chat = infix_processor(chat)
                if not isinstance(chat, Chat):
                    logger.debug(
                        "Infix function did not return a Chat object, indicating that no postprocessing should happen"
                    )
                    return chat

            if chat.is_latest_message_function_call:
                if not functions:
                    raise UnexpectedFunctionCallError(
                        f"Unexpected function call received from api: {chat.latest_message}"
                    )
                print("should call f")
                print(chat.latest_message)
                print()
                return "f call"
            else:
                if hasattr(signature.return_annotation, "model_json_schema"):
                    logger.debug(
                        "Looking for JSON documents in chat history, will attempt to deserialize into return type"
                    )
                    json_objects = chat.extract_json()
                    if json_objects:
                        for json_object in json_objects[::-1]:
                            try:
                                print("obj====================")
                                print()
                                print(json_object)
                                print("-==------------")
                                print()
                                return signature.return_annotation.model_validate_json(
                                    json_object
                                )
                            except ValidationError as error:
                                logger.debug(
                                    f"Found json in chat, but it did not conform to the return type. json: {json_object}\nReturn type: {signature.return_annotation}",
                                    exc_info=error,
                                )

                    logger.debug(
                        "No valid JSON found in chat, trying to generate valid JSON for return type with an additional message"
                    )
                    data = extract_json_gpt_35(signature.return_annotation, chat)
                    if not data:
                        data = extract_json_gpt_4(signature.return_annotation, chat)
                        if not data:
                            raise BadJsonResponseError("Failed to parse response JSON")
                    logger.debug(
                        "Successfully generated json that matches pydantic output type"
                    )
                    return data
                else:
                    return messages[-1].content

        return chat_complete

    return create_chat_completion
