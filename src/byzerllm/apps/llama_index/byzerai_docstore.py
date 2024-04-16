from typing import Dict, List, Optional, Sequence, Tuple

import json

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.utils import doc_to_json, json_to_doc
from llama_index.core.storage.kvstore.types import DEFAULT_BATCH_SIZE, BaseKVStore

from byzerllm.utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.langutil import asyncfy_with_semaphore
from byzerllm.apps.llama_index.simple_retrieval import SimpleRetrieval
from byzerllm.apps.llama_index.byzerai_kvstore import ByzerAIKVStore


class ByzerAIDocumentStore(KVDocumentStore):        

    def __init__(
        self,
        llm:ByzerLLM,
        retrieval:ByzerRetrieval,        
        namespace: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Init a KVDocumentStore."""
        
        self._llm = llm
        self._retrieval = SimpleRetrieval(llm=llm, retrieval=retrieval)         
        kv_store = ByzerAIKVStore(llm=llm, retrieval=retrieval)
        super().__init__(kv_store, namespace=namespace, batch_size=batch_size)          
