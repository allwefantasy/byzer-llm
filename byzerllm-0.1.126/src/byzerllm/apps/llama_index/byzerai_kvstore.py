import json
from typing import Any, Dict, List, Optional, Tuple, cast

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

from byzerllm.utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.apps.llama_index.simple_retrieval import SimpleRetrieval
from byzerllm.utils.langutil import asyncfy_with_semaphore

class ByzerAIKVStore(BaseKVStore):
   

    def __init__(
        self,
        llm:ByzerLLM,
        retrieval:ByzerRetrieval,                
        **kwargs: Any,
    ) -> None:
        self._llm = llm
        self._retrieval = SimpleRetrieval(llm=llm, retrieval=retrieval, **kwargs)

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """             
        self._retrieval.save_doc(data=[{            
            "doc_id":key,
            "json_data":json.dumps(val,ensure_ascii=False),
            "collection":collection,
            "content":"",
        }],owner=None)
        self._retrieval.commit_doc()

    async def aput(
        self, key: str, val: dict, collection: str = DEFAULT_COLLECTION
    ) -> None:
        """Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        raise asyncfy_with_semaphore(self.put)(key, val, collection)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Put a dictionary of key-value pairs into the store.

        Args:
            kv_pairs (List[Tuple[str, dict]]): key-value pairs
            collection (str): collection name

        """
        cur_batch = 0
        v = []        
        for key, val in kv_pairs:
            v.append({            
            "doc_id":key,
            "json_data":json.dumps(val,ensure_ascii=False),
            "collection":collection,
            "content":val.get("text",""),
        })
            cur_batch += 1

            if cur_batch >= batch_size:
                cur_batch = 0
                self._retrieval.save_doc(data=v,owner=None)
                v.clear()

        if cur_batch > 0:
            self._retrieval.save_doc(data=v,owner=None)

        self._retrieval.commit_doc()    
            

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        doc = self._retrieval.get_doc(doc_id=key,collection = collection)
        val_str = doc["json_data"] if doc else None
        if val_str is None:
            return None
        return json.loads(val_str)

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        return asyncfy_with_semaphore(self.get)(key, collection)

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        raise NotImplementedError

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """Get all values from the store."""
        raise NotImplementedError

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        self._retrieval.delete_doc(doc_id=key,collection = collection)
        return True

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        asyncfy_with_semaphore(self.delete)(key, collection)
    
