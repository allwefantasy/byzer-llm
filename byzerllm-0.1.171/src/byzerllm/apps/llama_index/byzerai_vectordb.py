import json
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, cast

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (    
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,    
    VectorStoreQueryResult,
)
from llama_index.core.schema import (
    BaseNode,
    ImageNode,
    IndexNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from byzerllm.utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.apps.llama_index.simple_retrieval import SimpleRetrieval

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

def metadata_dict_to_node(metadata: dict, text: Optional[str] = None) -> BaseNode:
    """Common logic for loading Node data from metadata dict."""
    node_json = metadata.get("_node_content", None)
    node_type = metadata.get("_node_type", None)
    if node_json is None:
        raise ValueError("Node content not found in metadata dict.")

    node: BaseNode
    if node_type == IndexNode.class_name():
        node = IndexNode.parse_raw(node_json)
    elif node_type == ImageNode.class_name():
        node = ImageNode.parse_raw(node_json)
    else:
        node = TextNode.parse_raw(node_json)

    if text is not None:
        node.set_content(text)

    return node

class ByzerAIVectorStore(VectorStore):    
    
    stores_text: bool = True       

    def __init__(
        self,
        llm:ByzerLLM,
        retrieval:ByzerRetrieval,
        chunk_collection: Optional[str] = "default",                                                   
        **kwargs: Any,
    ) -> None:        
        self._llm = llm
        self._retrieval = SimpleRetrieval(llm=llm, retrieval=retrieval,chunk_collection=chunk_collection,**kwargs)        
        

    @property
    def client(self) -> None:
        """Get client."""
        return

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        v = self._retrieval.get_chunk_by_id(text_id)
        if len(v) == 0:
            return []
                
        return v[0]["chunk_vector"] 

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index."""
        v = []
        count = 0
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
        self._retrieval.commit_chunk()
        
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.
        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """        
        chunks = self._retrieval.get_chunks_by_docid(ref_doc_id)        
        self._retrieval.delete_by_ids([chunk["_id"] for chunk in chunks])
        self._retrieval.commit_chunk()    
        

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
             
        
        query_embedding = cast(List[float], query.query_embedding)
        chunks = self._retrieval.search_content_chunks(owner="default",
                                              query_str=query.query_str,
                                              query_embedding=query_embedding,                                              
                                              doc_ids=query.node_ids,
                                              limit=100,
                                              return_json=False)
        
        chunks_map = {}
        for chunk in chunks:
            chunk["metadata"] = json.loads(chunk["json_data"])
            chunks_map[chunk["_id"]] = chunk["metadata"]
            
        query_filter_fn = _build_metadata_filter_fn(
            lambda node_id: chunks_map[node_id], query.filters
        )
        
        top_similarities = []
        top_ids = []
        
        counter = query.similarity_top_k
        nodes = []
        for chunk in chunks:
            if query_filter_fn(chunk["_id"]):                
                if counter <= 0:
                    break
                top_similarities.append(chunk["_score"])
                top_ids.append(chunk["_id"])
                try:
                    node = metadata_dict_to_node({"_node_content": chunk["metadata"]})
                    node.text = chunk["chunk"]
                except Exception:
                    # TODO: Legacy support for old metadata format
                    node = TextNode(
                        text=chunk["raw_chunk"],
                        id_=chunk["_id"],
                        embedding=None,
                        metadata=chunk["metadata"],                        
                        relationships={
                            NodeRelationship.SOURCE: RelatedNodeInfo(node_id=chunk["doc_id"])
                        },
                    )                
                nodes.append(node)
                counter -= 1

        return VectorStoreQueryResult(nodes = nodes ,similarities=top_similarities, ids=top_ids)


