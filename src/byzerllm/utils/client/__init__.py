from typing import List,Optional,Dict
from byzerllm.utils.client.types import (
    Templates,Template,Role,LLMHistoryItem,
    LLMRequest,
    LLMFunctionCallResponse,
    LLMClassResponse,InferBackend,EventName,EventCallbackResult,EventCallback,LLMResponse,FintuneRequestExtra,
    FintuneRequest,ExecuteCodeResponse,LLMMetadata
)
from byzerllm.utils.client.byzerllm_client import ByzerLLM

def default_chat_wrapper(llm:ByzerLLM,conversations: Optional[List[Dict]] = None,llm_config={}):
    return llm.chat_oai(conversations=conversations,llm_config=llm_config)

__all__ = [
    "ByzerLLM","default_chat_wrapper","Templates","Template","Role","LLMHistoryItem",
    "LLMRequest",
    "LLMFunctionCallResponse",
    "LLMClassResponse","InferBackend","EventName","EventCallbackResult","EventCallback","LLMResponse","FintuneRequestExtra",
    "FintuneRequest","ExecuteCodeResponse"]


