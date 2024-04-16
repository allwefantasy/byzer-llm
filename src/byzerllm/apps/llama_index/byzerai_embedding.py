"""Langchain Embedding Wrapper Module."""

from typing import TYPE_CHECKING, List, Optional
from llama_index.core.base.embeddings.base import DEFAULT_EMBED_BATCH_SIZE, BaseEmbedding
from llama_index.legacy.bridge.pydantic import PrivateAttr

from byzerllm.utils.client import ByzerLLM
from byzerllm.utils.langutil import asyncfy_with_semaphore

class ByzerAIEmbedding(BaseEmbedding):
    
    _llm: ByzerLLM = PrivateAttr()
    def __init__(
        self,
        llm:ByzerLLM,
    ):   
        self._llm = llm                     
        super().__init__(            
            model_name=self._llm.default_emb_model_name,
        )

    @classmethod
    def class_name(cls) -> str:
        return "ByzerAIEmbedding"    

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._llm.emb_query(query)[0].output[0:1024]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return asyncfy_with_semaphore(self._get_query_embedding, query)
            

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return asyncfy_with_semaphore(self._get_query_embedding, text)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        return [self._get_text_embedding(text) for text in texts]
