from byzerllm.utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.apps.llama_index.byzerai import ByzerAI
from byzerllm.apps.llama_index.byzerai_embedding import ByzerAIEmbedding
from byzerllm.apps.llama_index.byzerai_kvstore import ByzerAIKVStore
from byzerllm.apps.llama_index.byzerai_docstore import ByzerAIDocumentStore
from byzerllm.apps.llama_index.byzerai_index_store import ByzerAIIndexStore
from llama_index.service_context import ServiceContext
from llama_index.storage import StorageContext

def get_service_context(llm:ByzerLLM,**kargs):        
    service_context = ServiceContext.from_defaults(llm=ByzerAI(llm=llm),embed_model=ByzerAIEmbedding(llm=llm),**kargs)
    return service_context

def get_storage_context(llm:ByzerLLM,retrieval:ByzerRetrieval,**kargs):
    kvstore = ByzerAIKVStore(llm=llm, retrieval=retrieval)
    docstore = ByzerAIDocumentStore(llm=llm, retrieval=retrieval)
    index_store = ByzerAIIndexStore(llm=llm, retrieval=retrieval)
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        kvstore=kvstore,
        index_store=index_store,
        **kargs
    )
    return storage_context    
