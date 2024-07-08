try:
    from unittest.mock import patch
    def mock_nltk_data_find(resource_name, paths=None):
        return ""
    def mock_nltk_download(resource_name, download_dir=None):
        pass
    patcher_find = patch("nltk.data.find", side_effect=mock_nltk_data_find)
    patcher_download = patch("nltk.download", side_effect=mock_nltk_download)
    patcher_find.start()
    patcher_download.start()
except Exception as e:
    print("Error in patching nltk.download", e)
    pass


import os
from typing import Any, Callable, Dict, Optional, Sequence

try:
    from llama_index.core.bridge.pydantic import Field, PrivateAttr
except ImportError:
    from pydantic import Field, PrivateAttr

from llama_index.core.llms import (    
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.custom import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.utils import get_cache_dir
from byzerllm.utils.client import ByzerLLM

class ByzerAI(CustomLLM):
    """
    ByzerAI is a custom LLM that uses the ByzerLLM API to generate text.
    """    
   
    verbose: bool = Field(
        default=False,
        description="Whether to print verbose output.",
    )

    _model: ByzerLLM = PrivateAttr()

    def __init__(
        self,
        llm:ByzerLLM
    ) -> None:        
        self._model = llm                
        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        return "ByzerAI_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata.parse_obj(self._model.metadata.model_dump())

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        conversations = [{
            "role":message.role,
            "content":message.content
        } for message in messages]
        m = self._model.chat_oai(conversations=conversations)
        completion_response = CompletionResponse(text=m[0].output, raw=None)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:        
        conversations = [{
            "role":message.role,
            "content":message.content
        } for message in messages]
        m = self._model.stream_chat_oai(conversations=conversations)
        def gen():
            v = ""
            for response in m:                
                text:str = response[0]
                metadata:Dict[str,Any] = response[1]
                completion_response = CompletionResponse(text=text, delta=text[len(v):], raw=None)
                v = text
                yield completion_response
        return stream_completion_response_to_chat_response(gen())

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:        
        m = self._model.chat_oai(conversations=[{"role":"user","content":prompt}])
        completion_response = CompletionResponse(text=m[0].output, raw=None)
        return completion_response

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        conversations=[{"role":"user","content":prompt}]
        m = self._model.stream_chat_oai(conversations=conversations)
        def gen():
            v = ""
            for response in m:                
                text:str = response[0]
                metadata:Dict[str,Any] = response[1]
                completion_response = CompletionResponse(text=text, delta=text[len(v):], raw=None)
                v = text
                yield completion_response
        return gen()        