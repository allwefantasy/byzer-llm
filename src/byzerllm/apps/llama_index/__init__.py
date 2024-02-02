from byzerllm.utils.client import ByzerLLM
from byzerllm.apps.llama_index.byzerai import ByzerAI
from byzerllm.apps.llama_index.byzerai_embedding import ByzerAIEmbedding
from llama_index.service_context import ServiceContext

def get_service_context(llm:ByzerLLM,**kargs):
    service_context = ServiceContext.from_defaults(llm=ByzerAI(llm=llm),embed_model=ByzerAIEmbedding(llm=llm),**kargs)
    return service_context
