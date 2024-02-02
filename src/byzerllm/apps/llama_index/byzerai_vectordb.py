import json
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, cast

from llama_index.schema import BaseNode
from llama_index.utils import concat_dirs
from llama_index.vector_stores.types import (    
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,    
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import node_to_metadata_dict
from byzerllm.utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.retrieval.simple_retrieval import SimpleRetrieval

logger = logging.getLogger(__name__)


def _build_metadata_filter_fn(
    metadata_lookup_fn: Callable[[str], Mapping[str, Any]],
    metadata_filters: Optional[MetadataFilters] = None,
) -> Callable[[str], bool]:
    """Build metadata filter function."""
    filter_list = metadata_filters.legacy_filters() if metadata_filters else []
    if not filter_list:
        return lambda _: True

    def filter_fn(node_id: str) -> bool:
        metadata = metadata_lookup_fn(node_id)
        for filter_ in filter_list:
            metadata_value = metadata.get(filter_.key, None)
            if metadata_value is None:
                return False
            elif isinstance(metadata_value, list):
                if filter_.value not in metadata_value:
                    return False
            elif isinstance(metadata_value, (int, float, str, bool)):
                if metadata_value != filter_.value:
                    return False
        return True

    return filter_fn

class ByzerAIVectorStore(VectorStore):           

    def __init__(
        self,
        llm:ByzerLLM,
        retrieval:ByzerRetrieval,
        retrieval_cluster: str = "default",
        retrieval_db="default",                                            
        **kwargs: Any,
    ) -> None:        
        self._llm = llm
        self._retrieval = SimpleRetrieval(llm=llm, retrieval=retrieval, 
                                            retrieval_cluster=retrieval_cluster,
                                            retrieval_db=retrieval_db,                                            
                                          **kwargs)
        

    @classmethod
    def from_persist_dir(
        cls,
        llm:ByzerLLM,
        retrieval:ByzerRetrieval,
        retrieval_cluster: str,
        retrieval_db: str,
    ) -> "ByzerAIVectorStore":
        """Load from persist dir."""
        return cls(llm,retrieval,retrieval_cluster,retrieval_db)

    @classmethod
    def from_namespaced_persist_dir(
        cls,
        llm:ByzerLLM,
        retrieval:ByzerRetrieval,
        retrieval_cluster: str,
        retrieval_db: str,
    ) -> Dict[str, VectorStore]:
        return cls(llm,retrieval,retrieval_cluster,retrieval_db)
        

    @property
    def client(self) -> None:
        """Get client."""
        return

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        v = self._retrieval.get_chunk_by_id(text_id)
        return v["chunk_vector"] 

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index."""
        v = []
        for node in nodes:                        
            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=False
            )
            metadata.pop("_node_content", None)            
            m = {
                "chunk_id": node.node_id,
                "ref_doc_id": node.ref_doc_id,
                "metadata": node.metadata,
                "chunk_embedding": node.get_embedding(),
                "chunk_content": node.get_content(),
                "owner":""                
            }
            v.append(m)
        self._retrieval.save_chunks(v)    
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.
        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """        
        chunks = self._retrieval.get_chunks_by_docid(ref_doc_id)        
        self._retrieval.delete_by_ids([chunk["_id"] for chunk in chunks])    
        

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
             
        
        query_embedding = cast(List[float], query.query_embedding)
        chunks = self._retrieval.search_content_chunks(owner="",
                                              query_str=query.query_str,
                                              query_embedding=query_embedding,
                                              doc_ids=[],
                                              limit=4,
                                              return_json=False)
        chunks_map = {}
        for chunk in chunks:
            chunk["metadata"] = json.loads(chunk["metadata"])
            chunks_map[chunk["_id"]] = chunk["metadata"]
            
        query_filter_fn = _build_metadata_filter_fn(
            lambda node_id: chunks_map[node_id], query.filters
        )

        if query.node_ids is not None:
            available_ids = set(query.node_ids)

            def node_filter_fn(node_id: str) -> bool:
                return node_id in available_ids

        else:

            def node_filter_fn(node_id: str) -> bool:
                return True

        top_similarities = []
        top_ids = []

        for chunk in chunks:
            if query_filter_fn(chunk["_id"]) and node_filter_fn(chunk["_id"]):
                top_similarities.append(chunk["score"])
                top_ids.append(chunk["_id"])

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)


