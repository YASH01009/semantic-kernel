# Copyright (c) Yashwin Enamadi. All rights reserved.


from bardapi import Bard
from logging import Logger
from typing import Optional, Union, List

from semantic_kernel.connectors.ai.ai_exception import AIException
from semantic_kernel.connectors.ai.complete_request_settings import (
    CompleteRequestSettings,
)

from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.utils.null_logger import NullLogger


class BardTextCompletion(TextCompletionClientBase):
    _api_key: str
    _logger: Logger

    def __init__(
            self,
            api_key: str,
            log: Optional[Logger] = None,
    ):
        """
        Initializes a new instance of BardTextCompletion class.

        Arguments:
            api_key {str} -- Bard API key, see
                https://generativeai.pub/googles-bard-a-step-by-step-guide-to-using-the-unofficial-bard-api-3abb5b2d6abc or
                https://github.com/dsdanielpark/Bard-API
        """
        self._api_key = api_key
        self._logger = log if log is not None else NullLogger()

    async def complete_async(
        self, prompt: str, request_settings: CompleteRequestSettings = None
    ) -> Union[str, List[str]]:
        """
        Completes the given prompt. Returns a single string completion.

        Arguments:
            prompt {str} -- The prompt to complete.
            request_settings {CompleteRequestSettings} -- The request settings. This doesn't matter.

        Returns:
            str -- The completed text.
        """
        if not prompt:
            raise ValueError("The prompt cannot be `None` or empty")

        try:
            response = Bard(token=self._api_key).get_answer(input_text=prompt)
        except Exception as ex:
            raise AIException(
                AIException.ErrorCodes.ServiceError,
                "OpenAI service failed to complete the prompt",
                ex,
            )
        return response['content']

    async def complete_stream_async(self, prompt: str, settings: CompleteRequestSettings, logger: Logger):
        raise NotImplementedError()
