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
from byzerllm.utils.types import SingleOutputMeta, SingleOutput, StreamOutputs
from openai import OpenAI, AsyncOpenAI
from byzerllm.utils.client.mgenerai import MGeminiAI, MAsyncGeminiAI
import inspect
import functools
from byzerllm.utils.client.types import (
    Templates,
    Template,
    Role,
    LLMHistoryItem,
    LLMRequest,
    LLMFunctionCallResponse,
    LLMClassResponse,
    InferBackend,
    EventName,
    EventCallbackResult,
    EventCallback,
    LLMResponse,
    FintuneRequestExtra,
    FintuneRequest,
    ExecuteCodeResponse,
    LLMMetadata,
)

from byzerllm.utils import format_prompt_jinja2, format_str_jinja2

import pydantic
from pydantic import BaseModel
from loguru import logger
from byzerllm.utils.langutil import asyncfy_with_semaphore


class SimpleByzerLLM:
    """
    A simplified version of ByzerLLM that uses the OpenAI Python SDK
    for text/chat generation. The following methods mirror ByzerLLM's
    signatures but internally rely on openai.* calls.
    """

    def __init__(self, default_model_name: str = "deepseek_chat"):
        # You can store model names, placeholders, etc. in these variables:
        self.default_model_name = default_model_name
        # Keep track of "requests" or "deploy" info in a dict if needed
        self.deployments = {}
        self.sub_clients = {}
        self.event_callbacks: Dict[EventName, List[EventCallback]] = {}
        self.skip_nontext_check = False

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
            if not "saas.api_key" in infer_params:
                raise ValueError("saas.api_key is required for SaaS models")

            api_key = infer_params["saas.api_key"]
            model = infer_params.get("saas.model", "deepseek-chat")
            max_output_tokens = infer_params.get("saas.max_output_tokens", 8096)

            is_reasoning = infer_params.get(
                "is_reasoning", infer_params.get("saas.is_reasoning", False)
            )            
                
            if pretrained_model_type == "saas/gemini":
                sync_client = MGeminiAI(api_key=api_key, base_url=base_url)
                async_client = MAsyncGeminiAI(api_key=api_key, base_url=base_url)
            else:
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
                "is_reasoning": is_reasoning,
                "max_output_tokens": max_output_tokens,
            }
        else:
            raise ValueError(
                f"Unsupported pretrained_model_type: {pretrained_model_type}"
            )

        return {"model": udf_name, "status": "deployed"}

    @property
    def metadata(self) -> LLMMetadata:
        meta = self.get_meta(model=self.default_model_name)
        return LLMMetadata(
            context_window=meta.get("max_model_len", 8192),
            num_output=meta.get("num_output", 256),
            is_chat_model=not meta.get("embedding_mode", False),
            is_function_calling_model=True,
            model_name=meta.get("model_name", self.default_model_name),
        )

    def setup_default_model_name(self, model: str):
        self.default_model_name = model

    @staticmethod
    def from_default_model(
        model: str, auto_connect_cluster: bool = True
    ) -> "SimpleByzerLLM":
        llm = SimpleByzerLLM()
        llm.setup_default_model_name(model)
        return llm

    def is_model_exist(self, udf_name: str) -> bool:
        return udf_name in self.deployments

    def emb(self, model, request: LLMRequest, extract_params: Dict[str, Any] = {}):
        deploy_info = self.deployments[model or self.default_model_name]
        model_name = deploy_info["model"]
        client = deploy_info["sync_client"]
        resp = client.embeddings.create(input=[request.instruction], model=model_name)
        usage = resp.usage

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(usage, "prompt_tokens"):
            prompt_tokens = usage.prompt_tokens
        if hasattr(usage, "completion_tokens"):
            completion_tokens = usage.completion_tokens

        return [
            LLMResponse(
                output=resp.data[0].embedding,
                metadata={
                    "input_tokens_count": prompt_tokens,
                    "generated_tokens_count": completion_tokens,
                },
                input=request.instruction,
            )
        ]

    def emb_query(self, v: str, model: str = None):
        return self.emb(model=model, request=LLMRequest(instruction=v))

    def setup_sub_client(
        self,
        client_name: str,
        client: Union[List["SimpleByzerLLM"], "SimpleByzerLLM"] = None,
    ) -> "SimpleByzerLLM":
        if isinstance(client, list):
            self.sub_clients[client_name] = client
        else:
            self.sub_clients[client_name] = client
        return self

    def get_sub_client(
        self, client_name: str
    ) -> Union[List["SimpleByzerLLM"], Optional["SimpleByzerLLM"]]:
        return self.sub_clients.get(client_name, None)

    def remove_sub_client(self, client_name: str) -> "SimpleByzerLLM":
        if client_name in self.sub_clients:
            del self.sub_clients[client_name]
        return self

    def add_event_callback(
        self, event_name: EventName, callback: EventCallback
    ) -> None:
        self.event_callbacks.setdefault(event_name, []).append(callback)

    def _trigger_event(self, event_name: EventName, *args, **kwargs) -> Optional[Any]:
        if event_name in self.event_callbacks:
            for callback in self.event_callbacks[event_name]:
                continue_flag, value = callback(*args, **kwargs)
                if not continue_flag:
                    return value
        return None

    def process_messages(
        self, deploy_info: Dict[str, Any], messages: List[Dict[str, Any]], **kwargs
    ):
        extra_params = {}
        extra_body = {}
        base_url = deploy_info["infer_params"].get(
            "saas.base_url", "https://api.deepseek.com/v1"
        )
        if (
            len(messages) > 1
            and messages[-1]["role"] == "user"
            and messages[-2]["role"] == "user"
        ):
            messages[-1]["role"] = "assistant"

        def is_deepseek_chat_prefix():

            if kwargs.get("response_prefix", "false") in ["true", "True", True]:
                return True

            if messages[-1]["role"] == "assistant":
                if "deepseek" in base_url:
                    return True
            return False

        def is_siliconflow_chat_prefix():
            if messages[-1]["role"] == "assistant":
                if "siliconflow" in base_url:
                    return True
            return False

        if is_deepseek_chat_prefix():
            temp_message = {
                "role": "assistant",
                "content": messages[-1]["content"],
                "prefix": True,
            }
            logger.info(
                f"response_prefix is True, add prefix to the last message {temp_message['role']} {temp_message['content'][0:100]}"
            )
            messages = messages[:-1] + [temp_message]

        if is_siliconflow_chat_prefix():
            extra_body["prefix"] = messages[-1]["content"]
            extra_params["extra_body"] = extra_body
            messages = messages[:-1]

        if "stop" in kwargs:
            extra_params["stop"] = (
                kwargs["stop"]
                if isinstance(kwargs["stop"], list)
                else json.loads(kwargs["stop"])
            )

        return messages, extra_params

    def get_meta(self, model: str, llm_config: Dict[str, Any] = {}):
        """
        Return minimal metadata about the 'model'.
        We don't have direct metadata from OpenAI in real usage
        (some data can be gleaned from Model endpoint).
        """

        if model not in self.deployments:
            raise ValueError(f"Model {model} not deployed")

        deploy_info = self.deployments[model]

        # For SaaS models, get model name from deployment info
        model_name = deploy_info["model"]

        meta_result = {
            "model_name": model_name,
            "backend": "openai",
            "max_model_len": 8192,
            "support_stream": True,
            "deploy_info": deploy_info,
        }

        meta_result.update(
            {
                "model_deploy_type": "saas",
                "message_format": True,
                "support_chat_template": True,
            }
        )

        return meta_result

    def abort(self, request_id: str, model: Optional[str] = None):
        """
        ByzerLLM abort is used for vLLM streaming. For openai
        there's no direct 'abort' mechanism except manually
        canceling an HTTP request. We'll do no-op here.
        """
        # No actual abort logic in openai
        pass

    def process_audio(self, messages: Dict[str, Any]):
        return messages, False
        # if self.skip_nontext_check:
        #     return messages,False
        # try:
        #     from byzerllm.utils.nontext import Audio
        #     for message in messages:
        #         if message["role"] == "user":
        #             audio = Audio(message["content"])
        #             if audio.has_audio():
        #                 temp_content = audio.to_content()
        #                 final_content = []
        #                 for item in temp_content:
        #                     if "text" in item:
        #                         final_content.append({
        #                             "type":"text",
        #                             "text": item["text"]
        #                         })
        #                     if "audio" in item:
        #                         final_content.append({
        #                             "type":"audio",
        #                             "audio": item["audio"]
        #                         })
        #                 message["content"] = final_content
        # except Exception as inst:
        #     traceback.print_exc()
        #     logger.error(f"process_audio error: {inst}")
        #     return messages,False
        # return messages,True

    def process_image(self, messages: Dict[str, Any]):
        if self.skip_nontext_check:
            return messages, False
        try:
            from byzerllm.utils.nontext import Image

            for message in messages:
                if message["role"] == "user":
                    image = Image(message["content"])
                    if image.has_image():
                        temp_content = image.to_content()
                        final_content = []
                        for item in temp_content:
                            if "text" in item:
                                final_content.append(
                                    {"type": "text", "text": item["text"]}
                                )
                            if "image" in item:
                                final_content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": item["image"],
                                        },
                                    }
                                )
                        message["content"] = final_content
        except Exception as inst:
            traceback.print_exc()
            logger.error(f"process_image error: {inst}")
            return messages, False
        return messages, True

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
        extra_request_params: Dict[str, Any] = {},
    ) -> List[LLMResponse]:
        """
        This method mirrors ByzerLLM's chat_oai signature, but we implement
        it with OpenAI ChatCompletion calls. Summarily ignoring or stubbing
        some advanced ByzerLLM logic.
        """
        if not model:
            model = self.default_model_name

        # If only_return_prompt is True, we won't call openai, just return the prompt
        if only_return_prompt:
            return [
                LLMResponse(output="", metadata={}, input=conversations[-1]["content"])
            ]

        deploy_info = self.deployments.get(model, {})
        # logger.info(f"deploy_info: {deploy_info}")
        client = deploy_info["sync_client"]
        messages, extra_params = self.process_messages(
            deploy_info, conversations, **llm_config
        )

        is_reasoning = deploy_info["is_reasoning"]
        start_time = time.monotonic()

        history = messages[:-1]
        instruction = messages[-1]["content"]
        v = [{"instruction": instruction, "history": history, **llm_config}]

        event_result = self._trigger_event(EventName.BEFORE_CALL_MODEL, self, model, v)

        if event_result is not None:
            responses = [
                LLMResponse(
                    output=item["predict"],
                    metadata=item.get("metadata", {}),
                    input=item["input"],
                )
                for item in event_result
            ]
            return responses

        if extra_request_params:
            if "extra_body" in extra_params:
                extra_params["extra_body"] = {
                    **extra_params["extra_body"],
                    **extra_request_params,
                }
            else:
                extra_params["extra_body"] = extra_request_params

        messages, is_processed = self.process_image(messages=messages)
        if not is_processed:
            messages, is_processed = self.process_audio(messages=messages)

        if is_reasoning:
            response = client.chat.completions.create(
                messages=messages,
                model=deploy_info["model"],
                stream=False,
                extra_headers={
                    "HTTP-Referer": "https://auto-coder.chat", 
                    "X-Title": "auto-coder"
                },
                **extra_params,
            )
        else:
            response = client.chat.completions.create(
                messages=messages,
                model=deploy_info["model"],
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", deploy_info["max_output_tokens"]),
                top_p=llm_config.get("top_p", 0.9),
                stream=False,
                extra_headers={
                    "HTTP-Referer": "https://auto-coder.chat", 
                    "X-Title": "auto-coder"
                },
                **extra_params,
            )

        if hasattr(response, "error"):
            base_url = deploy_info["infer_params"].get("saas.base_url", "")
            model_name = deploy_info["model"]
            name = model
            raise Exception(
                f"name:{name} base_url:{base_url} model_name:{model_name} extra_params:{extra_params}  error:{response.error}"
            )

        generated_text = response.choices[0].message.content

        reasoning_text = ""
        if hasattr(response.choices[0].message, "reasoning_content"):
            reasoning_text = response.choices[0].message.reasoning_content or ""

        generated_tokens_count = response.usage.completion_tokens
        input_tokens_count = response.usage.prompt_tokens
        time_cost = time.monotonic() - start_time
        gen_meta = {
            "metadata": {
                "request_id": response.id,
                "input_tokens_count": input_tokens_count,
                "generated_tokens_count": generated_tokens_count,
                "time_cost": time_cost,
                "first_token_time": 0,
                "speed": float(generated_tokens_count) / time_cost,
                # Available options: stop, eos, length, tool_calls
                "finish_reason": response.choices[0].finish_reason,
                "reasoning_content": reasoning_text,
                **extra_params,
            }
        }
        return [
            LLMResponse(output=generated_text, metadata=gen_meta["metadata"], input="")
        ]

    def stream_chat_oai(
        self,
        conversations,
        model: Optional[str] = None,
        role_mapping=None,
        delta_mode: bool = False,
        llm_config: Dict[str, Any] = {},
        extra_request_params: Dict[str, Any] = {},
    ):
        """
        Provide a streaming interface. Yields chunk by chunk from the OpenAI
        streaming response. We keep the signature the same as ByzerLLM.
        """
        if not model:
            model = self.default_model_name

        deploy_info = self.deployments.get(model, {})
        # logger.info(f"deploy_info: {deploy_info}")

        client = deploy_info["sync_client"]
        is_reasoning = deploy_info["is_reasoning"]

        messages, extra_params = self.process_messages(
            deploy_info, conversations, **llm_config
        )

        if extra_request_params:
            if "extra_body" in extra_params:
                extra_params["extra_body"] = {
                    **extra_params["extra_body"],
                    **extra_request_params,
                }
            else:
                extra_params["extra_body"] = extra_request_params

        messages, is_processed = self.process_image(messages=messages)
        if not is_processed:
            messages, is_processed = self.process_audio(messages=messages)
        
        # logger.info(f"messages: {json.dumps(messages, indent=4,ensure_ascii=False)}")
        if is_reasoning:
            response = client.chat.completions.create(
                messages=messages,
                model=deploy_info["model"],
                stream=True,
                stream_options={"include_usage": True},
                extra_headers={
                    "HTTP-Referer": "https://auto-coder.chat", 
                    "X-Title": "auto-coder"
                },
                **extra_params,
            )

        else:
            response = client.chat.completions.create(
                messages=messages,
                model=deploy_info["model"],
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", deploy_info["max_output_tokens"]),
                top_p=llm_config.get("top_p", 0.9),
                stream=True,
                stream_options={"include_usage": True},
                **extra_params,
            )

        if hasattr(response, "error"):
            base_url = deploy_info["infer_params"].get("saas.base_url", "")
            model_name = deploy_info["model"]
            name = model
            raise Exception(
                f"name:{name} base_url:{base_url} model_name:{model_name} extra_params:{extra_params}  error:{response.error}"
            )

        input_tokens_count = 0
        generated_tokens_count = 0
        last_meta = None
        if delta_mode:
            for chunk in response:

                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens_count = chunk.usage.prompt_tokens
                    generated_tokens_count = chunk.usage.completion_tokens
                else:
                    input_tokens_count = 0
                    generated_tokens_count = 0

                if not chunk.choices:
                    if last_meta:
                        yield (
                            "",
                            SingleOutputMeta(
                                input_tokens_count=input_tokens_count,
                                generated_tokens_count=generated_tokens_count,
                                reasoning_content="",
                                finish_reason=last_meta.finish_reason,
                            ),
                        )
                    continue

                content = chunk.choices[0].delta.content or ""

                reasoning_text = ""
                if hasattr(chunk.choices[0].delta, "reasoning_content"):
                    reasoning_text = chunk.choices[0].delta.reasoning_content or ""

                last_meta = SingleOutputMeta(
                    input_tokens_count=input_tokens_count,
                    generated_tokens_count=generated_tokens_count,
                    reasoning_content=reasoning_text,
                    finish_reason=chunk.choices[0].finish_reason,
                )
                yield (content, last_meta)
        else:
            s = ""
            all_reasoning_text = ""
            for chunk in response:

                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens_count = chunk.usage.prompt_tokens
                    generated_tokens_count = chunk.usage.completion_tokens
                else:
                    input_tokens_count = 0
                    generated_tokens_count = 0

                if not chunk.choices:
                    if last_meta:
                        yield (
                            s,
                            SingleOutputMeta(
                                input_tokens_count=input_tokens_count,
                                generated_tokens_count=generated_tokens_count,
                                reasoning_content=all_reasoning_text,
                                finish_reason=last_meta.finish_reason,
                            ),
                        )
                    continue

                content = chunk.choices[0].delta.content or ""
                reasoning_text = ""
                if hasattr(chunk.choices[0].delta, "reasoning_content"):
                    reasoning_text = chunk.choices[0].delta.reasoning_content or ""

                s += content
                all_reasoning_text += reasoning_text
                yield (
                    s,
                    SingleOutputMeta(
                        input_tokens_count=input_tokens_count,
                        generated_tokens_count=generated_tokens_count,
                        reasoning_content=all_reasoning_text,
                        finish_reason=chunk.choices[0].finish_reason,
                    ),
                )

    async def async_stream_chat_oai(
        self,
        conversations,
        role_mapping=None,
        model: Optional[str] = None,
        delta_mode: bool = False,
        llm_config: Dict[str, Any] = {},
        extra_request_params: Dict[str, Any] = {},
    ):
        if not model:
            model = self.default_model_name

        deploy_info = self.deployments.get(model, {})
        # logger.info(f"deploy_info: {deploy_info}")
        client = deploy_info["async_client"]
        is_reasoning = deploy_info["is_reasoning"]
        messages, extra_params = self.process_messages(
            deploy_info, conversations, **llm_config
        )

        logger.info(f"extra_params: {extra_params}")
        if extra_request_params:
            if "extra_body" in extra_params:
                extra_params["extra_body"] = {
                    **extra_params["extra_body"],
                    **extra_request_params,
                }
            else:
                extra_params["extra_body"] = extra_request_params

        messages, is_processed = self.process_image(messages=messages)
        if not is_processed:
            messages, is_processed = self.process_audio(messages=messages)

        # logger.info(f"messages: {json.dumps(messages, indent=4,ensure_ascii=False)}") 
        if is_reasoning:
            response = await client.chat.completions.create(
                messages=messages,
                model=deploy_info["model"],
                stream=True,
                stream_options={"include_usage": True},
                extra_headers={
                    "HTTP-Referer": "https://auto-coder.chat", 
                    "X-Title": "auto-coder"
                },
                **extra_params,
            )
        else:
            response = await client.chat.completions.create(
                messages=messages,
                model=deploy_info["model"],
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", deploy_info["max_output_tokens"]),
                top_p=llm_config.get("top_p", 0.9),
                stream=True,
                stream_options={"include_usage": True},
                **extra_params,
            )

        if hasattr(response, "error"):
            base_url = deploy_info["infer_params"].get("saas.base_url", "")
            model_name = deploy_info["model"]
            name = model
            raise Exception(
                f"name:{name} base_url:{base_url} model_name:{model_name} extra_params:{extra_params}  error:{response.error}"
            )

        input_tokens_count = 0
        generated_tokens_count = 0
        last_meta = None
        if delta_mode:
            async for chunk in response:
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens_count = chunk.usage.prompt_tokens
                    generated_tokens_count = chunk.usage.completion_tokens
                else:
                    input_tokens_count = 0
                    generated_tokens_count = 0

                if not chunk.choices:
                    if last_meta:
                        yield (
                            "",
                            SingleOutputMeta(
                                input_tokens_count=input_tokens_count,
                                generated_tokens_count=generated_tokens_count,
                                reasoning_content="",
                                finish_reason=last_meta.finish_reason,
                            ),
                        )
                    continue

                content = chunk.choices[0].delta.content or ""
                reasoning_text = ""
                if hasattr(chunk.choices[0].delta, "reasoning_content"):
                    reasoning_text = chunk.choices[0].delta.reasoning_content or ""

                last_meta = SingleOutputMeta(
                    input_tokens_count=input_tokens_count,
                    generated_tokens_count=generated_tokens_count,
                    reasoning_content=reasoning_text,
                    finish_reason=chunk.choices[0].finish_reason,
                )
                yield (content, last_meta)
        else:
            s = ""
            all_reasoning_text = ""
            async for chunk in response:

                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens_count = chunk.usage.prompt_tokens
                    generated_tokens_count = chunk.usage.completion_tokens
                else:
                    input_tokens_count = 0
                    generated_tokens_count = 0

                if not chunk.choices:
                    if last_meta:
                        yield (
                            s,
                            SingleOutputMeta(
                                input_tokens_count=input_tokens_count,
                                generated_tokens_count=generated_tokens_count,
                                reasoning_content=all_reasoning_text,
                                finish_reason=last_meta.finish_reason,
                            ),
                        )
                    continue

                content = chunk.choices[0].delta.content or ""
                reasoning_text = ""
                if hasattr(chunk.choices[0].delta, "reasoning_content"):
                    reasoning_text = chunk.choices[0].delta.reasoning_content or ""

                s += content
                all_reasoning_text += reasoning_text
                last_meta = SingleOutputMeta(
                    input_tokens_count=input_tokens_count,
                    generated_tokens_count=generated_tokens_count,
                    reasoning_content=all_reasoning_text,
                    finish_reason=chunk.choices[0].finish_reason,
                )
                yield (s, last_meta)

    def prompt(
        self,
        model: Optional[str] = None,
        render: Optional[str] = "jinja2",
        check_result: bool = False,
        options: Dict[str, Any] = {},
        return_origin_response: bool = False,
        marker: Optional[str] = None,
        assistant_prefix: Optional[str] = None,
        meta_holder: Optional[Any] = None,
        conversation: List[Dict[str,Any]] = []
    ):
        if model is None:
            if "model" in options:
                model = options.pop("model")
            else:
                model = self.default_model_name

        def is_instance_of_generator(v):
            from typing import Generator, get_origin, get_args
            import collections

            if get_origin(v) is collections.abc.Generator:
                args = get_args(v)
                if args == (str, type(None), type(None)):
                    return True
            return False

        def _impl(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                signature = inspect.signature(func)
                arguments = signature.bind(*args, **kwargs)
                arguments.apply_defaults()
                input_dict = {}
                for param in signature.parameters:
                    input_dict.update({param: arguments.arguments[param]})

                if "self" in input_dict:
                    instance = input_dict.pop("self")
                    new_input_dic = func(instance, **input_dict)
                    if new_input_dic and not isinstance(new_input_dic, dict):
                        raise TypeError(
                            f"Return value of {func.__name__} should be a dict"
                        )
                    if new_input_dic:
                        input_dict = {**input_dict, **new_input_dic}
                else:
                    new_input_dic = func(**input_dict)
                    if new_input_dic and not isinstance(new_input_dic, dict):
                        raise TypeError(
                            f"Return value of {func.__name__} should be a dict"
                        )
                    if new_input_dic:
                        input_dict = {**input_dict, **new_input_dic}

                prompt_str = format_prompt_jinja2(func, **input_dict)

                if marker:
                    prompt_str = f"{prompt_str}\n\n{marker}"

                if is_instance_of_generator(signature.return_annotation):
                    temp_options = {**{"delta_mode": True}, **options}
                    conversations = [{"role": "user", "content": prompt_str}]
                    if assistant_prefix:
                        conversations = conversations + [
                            {"role": "assistant", "content": assistant_prefix}
                        ]
                    
                    
                    t = self.stream_chat_oai(
                        conversations=conversations,
                        model=model,
                        **temp_options,
                    )

                    def generator():
                        for item, meta in t:
                            if meta_holder and meta:
                                meta_holder.meta = meta
                            yield item

                    if return_origin_response:
                        return t

                    return generator()

                if issubclass(signature.return_annotation, pydantic.BaseModel):
                    response_class = signature.return_annotation
                    conversations = [{"role": "user", "content": prompt_str}]
                    if assistant_prefix:
                        conversations = conversations + [
                            {"role": "assistant", "content": assistant_prefix}
                        ]
                    t = self.chat_oai(
                        model=model,
                        conversations=conversations,
                        response_class=response_class,
                        impl_func_params=input_dict,
                        **options,
                    )

                    if meta_holder and t[0].metadata:
                        meta_holder.meta = t[0].metadata

                    if return_origin_response:
                        return t
                    r: LLMClassResponse = t[0]
                    if r.value is None and check_result:
                        logger.warning(
                            f"""
                                {func.__name__} return None.
                                metadata:
                                {r.metadata}
                                response:
                                {r.response}
                            """
                        )
                    return r.value
                elif issubclass(signature.return_annotation, str):
                    conversations = [{"role": "user", "content": prompt_str}]
                    if assistant_prefix:
                        conversations = conversations + [
                            {"role": "assistant", "content": assistant_prefix}
                        ]
                    t = self.chat_oai(
                        model=model,
                        conversations=conversations,
                        **options,
                    )

                    if meta_holder and t[0].metadata:
                        meta_holder.meta = t[0].metadata

                    if return_origin_response:
                        return t
                    return t[0].output
                else:
                    raise Exception(
                        f"{func.__name__} should return a pydantic model or string"
                    )

            return wrapper

        return _impl
