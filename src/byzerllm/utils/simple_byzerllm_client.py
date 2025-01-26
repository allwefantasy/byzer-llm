
from typing import Optional, Dict, List, Union, Any
import json
from byzerllm.utils.client.types import LLMResponse, LLMMetadata
from byzerllm.saas.official_openai import CustomSaasAPI

# 全局 meta 缓存
_global_meta_cache = {}

class SimpleByzerLLM:
    def __init__(self, url: Optional[str] = None, **kwargs):
        self.url = url
        self.model = kwargs.get("model", "deepseek-chat")
        self.api_key = kwargs.get("api_key")
        self.client = None
        self.meta_cache = _global_meta_cache

    def undeploy(self, udf_name: str, force: bool = False):
        """Undeploy model"""
        self.client = None
        if udf_name in _global_meta_cache:
            del _global_meta_cache[udf_name]

    def deploy(self, model_path: str, pretrained_model_type: str, udf_name: str, infer_params: Dict[str, Any]):
        """Deploy model"""
        if not self.api_key:
            raise ValueError("API key is required")
        infer_params["saas.api_key"] = self.api_key
        infer_params["saas.model"] = self.model
        self.client = CustomSaasAPI(infer_params)
        return self.get_meta(udf_name)

    def get_meta(self, model: str, llm_config: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Get model metadata"""
        if model in _global_meta_cache:
            return _global_meta_cache[model]
        
        if not self.client:
            raise ValueError("Model not deployed")
            
        meta = self.client.get_meta()[0]
        _global_meta_cache[model] = meta
        return meta

    def abort(self, request_id: str, model: Optional[str] = None):
        """Abort request"""
        # OpenAI API does not support aborting requests
        pass

    def chat_oai(self, conversations, model: Optional[str] = None, **kwargs) -> List[LLMResponse]:
        """Chat with model"""
        if not self.client:
            raise ValueError("Model not deployed")
            
        if isinstance(conversations, str):
            conversations = [{"role": "user", "content": conversations}]
            
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in conversations]
        
        response = self.client.async_stream_chat(
            tokenizer=None,
            ins=messages[-1]["content"],
            his=messages[:-1],
            **kwargs
        )
        
        return [LLMResponse(output=response[0][0], metadata=response[0][1])]

    def stream_chat_oai(self, conversations, model: Optional[str] = None, **kwargs):
        """Stream chat with model"""
        if not self.client:
            raise ValueError("Model not deployed")
            
        if isinstance(conversations, str):
            conversations = [{"role": "user", "content": conversations}]
            
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in conversations]
        
        for chunk in self.client.async_stream_chat(
            tokenizer=None,
            ins=messages[-1]["content"],
            his=messages[:-1],
            **kwargs
        ):
            yield chunk

    async def async_stream_chat_oai(self, conversations, model: Optional[str] = None, **kwargs):
        """Async stream chat with model"""
        if not self.client:
            raise ValueError("Model not deployed")
            
        if isinstance(conversations, str):
            conversations = [{"role": "user", "content": conversations}]
            
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in conversations]
        
        async for chunk in self.client.async_stream_chat(
            tokenizer=None,
            ins=messages[-1]["content"],
            his=messages[:-1],
            **kwargs
        ):
            yield chunk