import os
import json
import asyncio
import openai
from typing import (
    Dict,
    Any,
    List,
    Optional,
    Union,
    Tuple,
    Callable,
)
from enum import Enum
import time
import traceback
from byzerllm.utils.types import LLMResponse,SingleOutputMeta,SingleOutput,StreamOutputs
from openai import OpenAI, AsyncOpenAI
class SimpleByzerLLM:
    """
    A simplified version of ByzerLLM that uses the OpenAI Python SDK
    for text/chat generation. The following methods mirror ByzerLLM's
    signatures but internally rely on openai.* calls.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        :param api_key: your openai api key. If None, tries environment variable.
        :param kwargs: any extra config if needed
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        openai.api_key = self.api_key
        # You can store model names, placeholders, etc. in these variables:
        self.default_model_name = kwargs.get("default_model_name", "deepseek_chat")
        # Keep track of "requests" or "deploy" info in a dict if needed
        self.deployments = {}

    def undeploy(self, udf_name: str, force: bool = False):
        """
        Mirror of ByzerLLM. Actually no-op in OpenAI usage. We just remove
        from local dictionary of 'deployed' models if any.
        """
        if udf_name in self.deployments:
            # Clean up OpenAI clients if they exist
            deploy_info = self.deployments[udf_name]
            if "sync_client" in deploy_info:
                deploy_info["sync_client"].close()
            if "async_client" in deploy_info:
                asyncio.run(deploy_info["async_client"].close())
            del self.deployments[udf_name]

    def deploy(
        self,
        model_path: str,
        pretrained_model_type: str,
        udf_name: str,
        infer_params: Dict[str, Any],
    ):
        """
        For local/hosted model we had path. For OpenAI usage,
        we might store a mapping from udf_name -> model_name.
        """
        # Initialize OpenAI clients if this is a SaaS model
        if pretrained_model_type.startswith("saas/"):
            base_url = infer_params.get("saas.base_url", "https://api.deepseek.com/v1")
            api_key = infer_params.get("saas.api_key", self.api_key)
            model = infer_params.get("saas.model", "deepseek-chat")
            
            # Create both sync and async clients
            sync_client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            
            async_client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            
            self.deployments[udf_name] = {
                "model_path": model_path,
                "pretrained_model_type": pretrained_model_type,
                "infer_params": infer_params,
                "sync_client": sync_client,
                "async_client": async_client,
                "model": model,
            }
        else:
            raise ValueError(f"Unsupported pretrained_model_type: {pretrained_model_type}")
            
        return {"model": udf_name,"status": "deployed"}

    def get_meta(self, model: str, llm_config: Dict[str, Any] = {}):
        """
        Return minimal metadata about the 'model'.
        We don't have direct metadata from OpenAI in real usage
        (some data can be gleaned from Model endpoint).
        """
        deploy_info = self.deployments.get(model, {})
        
        # For SaaS models, get model name from deployment info
        model_name = deploy_info.get("model", model)
        
        meta_result = {
            "model_name": model_name,
            "backend": "openai",
            "max_model_len": 4097,  
            "support_stream": True,
            "deploy_info": deploy_info,
        }
        
        # Add SaaS specific metadata if available
        if deploy_info.get("pretrained_model_type", "").startswith("saas/"):
            meta_result.update({
                "model_deploy_type": "saas",
                "message_format": True,
                "support_chat_template": True,
            })
            
        return meta_result

    def abort(self, request_id: str, model: Optional[str] = None):
        """
        ByzerLLM abort is used for vLLM streaming. For openai
        there's no direct 'abort' mechanism except manually
        canceling an HTTP request. We'll do no-op here.
        """
        # No actual abort logic in openai
        pass

    def chat_oai(
        self,
        conversations,
        tools: List[Union[Callable, str]] = [],
        tool_choice: Optional[Union[Callable, str]] = None,
        execute_tool: bool = False,
        impl_func: Optional[Callable] = None,
        execute_impl_func: bool = False,
        impl_func_params: Optional[Dict[str, Any]] = None,
        func_params: Optional[Dict[str, Any]] = None,
        response_class: Optional[Union[Any, str]] = None,
        response_after_chat: Optional[Union[Any, str]] = False,
        enable_default_sys_message: bool = True,
        model: Optional[str] = None,
        role_mapping=None,
        llm_config: Dict[str, Any] = {},
        only_return_prompt: bool = False,
    ) -> List[LLMResponse]:
        """
        This method mirrors ByzerLLM's chat_oai signature, but we implement
        it with OpenAI ChatCompletion calls. Summarily ignoring or stubbing
        some advanced ByzerLLM logic.
        """
        if not model:
            model = self.default_model_name

        # Convert the 'conversations' into the OpenAI Chat style messages
        # e.g. [ { "role": "user", "content": "..."} ]
        # For simplicity, let's assume 'conversations' is a list of dict: role, content.
        # Possibly user wants "system" or "user" or "assistant" roles.
        openai_messages = []
        for item in conversations:
            openai_messages.append({
                "role": item.get("role", "user"),
                "content": item.get("content", ""),
            })

        # If only_return_prompt is True, we won't call openai, just return the prompt
        if only_return_prompt:
            return [LLMResponse(output="(prompt only)", metadata={}, input=str(openai_messages))]

        # Merge any extra config from llm_config
        # Typically can handle temperature, top_p, max_tokens, etc.
        openai_params = {
            "model": model,
            "messages": openai_messages,
            "temperature": llm_config.get("temperature", 0.7),
            "max_tokens": llm_config.get("max_tokens", 4096),
            "top_p": llm_config.get("top_p", 1.0),
        }
        try:
            deploy_info = self.deployments.get(model, {})
            client = deploy_info.get("sync_client", openai)
            completion = client.chat.completions.create(**openai_params)
            content = completion.choices[0].message["content"]
            return [LLMResponse(output=content, metadata={"usage": completion.usage}, input=str(openai_messages))]
        except Exception as e:
            traceback_str = traceback.format_exc()
            return [LLMResponse(output=str(e), metadata={"traceback": traceback_str}, input=str(openai_messages))]

    def stream_chat_oai(
        self,
        conversations,
        model: Optional[str] = None,
        role_mapping=None,
        delta_mode: bool = False,
        llm_config: Dict[str, Any] = {},
    ):
        """
        Provide a streaming interface. Yields chunk by chunk from the OpenAI
        streaming response. We keep the signature the same as ByzerLLM.
        """
        if not model:
            model = self.default_model_name

        openai_messages = []
        for item in conversations:
            openai_messages.append({
                "role": item.get("role", "user"),
                "content": item.get("content", ""),
            })

        openai_params = {
            "model": model,
            "messages": openai_messages,
            "temperature": llm_config.get("temperature", 0.7),
            "max_tokens": llm_config.get("max_tokens", 256),
            "top_p": llm_config.get("top_p", 1.0),
            "stream": True,  # Turn on streaming
        }

        try:
            deploy_info = self.deployments.get(model, {})
            client = deploy_info.get("sync_client", openai)
            deploy_info = self.deployments.get(model, {})
            client = deploy_info.get("async_client", openai)
            stream = await client.chat.completions.create(**openai_params)
            collected_text = ""
            for chunk in stream:
                delta = chunk.choices[0]["delta"].get("content", "")
                if not delta:
                    continue
                collected_text += delta
                # For Byzer style: yield (text, metadata)
                yield (delta, {"all_text": collected_text})
        except Exception as e:
            # On error, we yield a final chunk with error message
            yield ("", {"error": str(e)})

    async def async_stream_chat_oai(
        self,
        conversations,
        role_mapping=None,
        model: Optional[str] = None,
        delta_mode: bool = False,
        llm_config: Dict[str, Any] = {},
    ):
        """
        Asynchronous generator version of stream_chat_oai.
        OpenAI's official streaming doesn't provide direct async interface in old sdk.
        If using a brand new openai python library that supports async, do so. Otherwise
        we'd need an httpx-based approach or wrap sync in thread.
        """
        # Fallback: wrap the sync method in a thread. Not truly async but signature is preserved.
        loop = asyncio.get_running_loop()

        def sync_generator():
            return self.stream_chat_oai(
                conversations=conversations,
                model=model,
                role_mapping=role_mapping,
                delta_mode=delta_mode,
                llm_config=llm_config,
            )

        # We'll iterate in a thread pool, yield from it
        it = await loop.run_in_executor(None, lambda: list(sync_generator()))
        for chunk, meta in it:
            yield (chunk, meta)
