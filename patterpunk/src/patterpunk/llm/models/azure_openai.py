from abc import ABC
from typing import List, Optional

from patterpunk.config.providers.azure_openai import (
    azure_openai,
    AZURE_OPENAI_MAX_RETRIES,
)
from patterpunk.llm.models.openai import OpenAiModel, OpenAiApiError
from patterpunk.llm.thinking import ThinkingConfig
from patterpunk.logger import logger

if azure_openai:
    from openai import APIError


class AzureOpenAiMissingConfigurationError(Exception):
    pass


class AzureOpenAiModel(OpenAiModel, ABC):
    def __init__(
        self,
        deployment_name="",
        temperature=None,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        logit_bias=None,
        thinking_config: Optional[ThinkingConfig] = None,
    ):
        if not azure_openai:
            raise AzureOpenAiMissingConfigurationError(
                "Azure OpenAI was not initialized correctly. "
                "Check that PP_AZURE_OPENAI_ENDPOINT and PP_AZURE_OPENAI_API_KEY are set."
            )

        super().__init__(
            model=deployment_name,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            thinking_config=thinking_config,
        )

    def _execute_with_retry(self, responses_parameters: dict) -> object:
        retry_count = 0
        done = False
        response = False

        while not done and retry_count < AZURE_OPENAI_MAX_RETRIES:
            try:
                response = azure_openai.responses.create(**responses_parameters)
                logger.info("Azure OpenAI Responses API response received")
                done = True
            except APIError as error:
                if (
                    "reasoning.summary" in str(error)
                    and "reasoning" in responses_parameters
                ):
                    logger.info(
                        "Organization not verified for reasoning summaries, removing reasoning parameter and treating as regular model"
                    )
                    responses_parameters.pop("reasoning", None)
                    responses_parameters["temperature"] = self.temperature
                    responses_parameters["top_p"] = self.top_p
                else:
                    logger.info(
                        "Retrying Azure OpenAI Responses API request due to APIError",
                        exc_info=error,
                    )
                retry_count += 1

        if not done or not response:
            raise OpenAiApiError(
                "Azure OpenAI Responses API is returning too many api errors"
            )

        return response

    def _log_request_start(self, messages: List) -> None:
        logger.info("Request to Azure OpenAI made")
        from patterpunk.logger import logger_llm

        logger_llm.debug(
            "\n---\n".join(
                [f"{message.__repr__(truncate=False)}" for message in messages]
            )
        )

    def _log_request_parameters(self, responses_parameters: dict) -> None:
        from patterpunk.logger import logger_llm

        log_params = {k: v for k, v in responses_parameters.items() if k != "input"}
        param_strings = []
        for key, value in log_params.items():
            param_strings.append(f"{key}: {value}")
        logger_llm.info(
            f"Azure OpenAI Responses API params: {', '.join(param_strings)}"
        )

    @staticmethod
    def get_name():
        return "Azure OpenAI"

    @staticmethod
    def get_available_models() -> List[str]:
        # Azure uses deployment names which are user-specific
        # We cannot list them via API without additional permissions
        return []
