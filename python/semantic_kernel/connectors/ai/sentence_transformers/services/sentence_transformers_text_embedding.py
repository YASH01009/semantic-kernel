# Copyright (c) Microsoft. All rights reserved.

from logging import Logger
from typing import Any, List, Optional

from numpy import array, ndarray
from semantic_kernel.connectors.ai.ai_exception import AIException

from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import EmbeddingGeneratorBase
from semantic_kernel.utils.null_logger import NullLogger
from sentence_transformers import SentenceTransformer


class SentenceTransformersTextEmbedding(EmbeddingGeneratorBase):
    _model_id: str
    _logger: Logger

    def __init__(
        self,
        model_id: str,
        log: Optional[Logger] = None
    ) -> None:
        """
        Initializes a new instance of the SentenceTransformersTextEmbedding class.

        Arguments:
            model_id {str} -- SentenceTranformers model name, see
                https://pypi.org/project/sentence-transformers/
                https://www.sbert.net/docs/pretrained_models.html
        """
        self._model_id = model_id
        self._log = log if log is not None else NullLogger()

    async def generate_embeddings_async(self, texts: List[str]) -> ndarray:
        try:
            model = SentenceTransformer(self._model_id)

            # make numpy arrays from the response
            raw_embeddings = model.encode(texts)
            return raw_embeddings
        except Exception as ex:
            raise AIException(
                AIException.ErrorCodes.ServiceError,
                "SentenceTransformers service failed to generate embeddings",
                ex,
            )
