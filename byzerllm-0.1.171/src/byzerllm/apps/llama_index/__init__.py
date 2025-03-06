from byzerllm.utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.apps.llama_index.byzerai import ByzerAI
from byzerllm.apps.llama_index.byzerai_embedding import ByzerAIEmbedding
from byzerllm.apps.llama_index.byzerai_docstore import ByzerAIDocumentStore
from byzerllm.apps.llama_index.byzerai_index_store import ByzerAIIndexStore
from byzerllm.apps.llama_index.byzerai_vectordb import ByzerAIVectorStore
from llama_index.core.service_context import ServiceContext
from llama_index.core.storage import StorageContext
from typing import Optional

def get_service_context(llm:ByzerLLM,**kargs):        
    service_context = ServiceContext.from_defaults(llm=ByzerAI(llm=llm),embed_model=ByzerAIEmbedding(llm=llm),**kargs)
    return service_context

def get_storage_context(llm:ByzerLLM,retrieval:ByzerRetrieval,
                        chunk_collection:Optional[str]="default",
                        namespace:Optional[str]=None,                        
                        **kargs):
    vector_store = ByzerAIVectorStore(llm=llm, retrieval=retrieval,chunk_collection=chunk_collection)
    docstore = ByzerAIDocumentStore(llm=llm, retrieval=retrieval,namespace=namespace)
    index_store = ByzerAIIndexStore(llm=llm, retrieval=retrieval,namespace=namespace)
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
        index_store=index_store,
        **kargs
    )
    return storage_context    
