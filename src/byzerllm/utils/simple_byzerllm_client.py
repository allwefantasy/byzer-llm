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
            base_url = infer_params.get(
                "saas.base_url", "https://api.deepseek.com/v1")
            api_key = infer_params.get("saas.api_key", self.api_key)
            model = infer_params.get("saas.model", "deepseek-chat")

            is_reasoning = infer_params.get("is_reasoning", False)

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
                "is_reasoning": is_reasoning,
            }
        else:
            raise ValueError(
                f"Unsupported pretrained_model_type: {pretrained_model_type}")

        return {"model": udf_name, "status": "deployed"}
    
    def process_messages(self, messages: List[Dict[str, Any]]):
        extra_params = {}
        extra_body = {}
        messages = [
            {"role": message["role"],
                "content": self.process_input(message["content"])}
            for message in his
        ] + [{"role": "user", "content": self.process_input(ins)}]

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
                if "deepseek" in self.other_params.get("base_url", ""):
                    return True
            return False

        def is_siliconflow_chat_prefix():
            if messages[-1]["role"] == "assistant":
                if "siliconflow" in self.other_params.get("base_url", ""):
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
            "max_model_len": 4097,
            "support_stream": True,
            "deploy_info": deploy_info,
        }

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

        # If only_return_prompt is True, we won't call openai, just return the prompt
        if only_return_prompt:
            return [LLMResponse(output="", metadata={}, input=conversations[-1]["content"])]

        deploy_info = self.deployments.get(model, {})
        client = deploy_info["sync_client"]
        
        messages, extra_params = self.process_messages(conversations)

        start_time = time.monotonic()
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 4096),
            top_p=llm_config.get("top_p", 1.0),
            **extra_params,
        )
        generated_text = response.choices[0].message.content
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
                **extra_params,
            }
        }
        return [LLMResponse(output=generated_text, metadata=gen_meta["metadata"], input="")]

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

        deploy_info = self.deployments.get(model, {})
        client = deploy_info["sync_client"]
        is_reasoning = deploy_info["is_reasoning"]

        messages, extra_params = self.process_messages(conversations)

        if is_reasoning:
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                stream=True,
                **extra_params,
            )
        else:
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 4096),
                top_p=llm_config.get("top_p", 1.0),
                stream=True,
                **extra_params,
            )
        input_tokens_count = 0
        generated_tokens_count = 0
        if delta_mode:
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens_count = chunk.usage.prompt_tokens
                    generated_tokens_count = chunk.usage.completion_tokens
                else:
                    input_tokens_count = 0
                    generated_tokens_count = 0

                yield (content, SingleOutputMeta(input_tokens_count=input_tokens_count, generated_tokens_count=generated_tokens_count, finish_reason=chunk.choices[0].finish_reason))
        else:
            s = ""
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens_count = chunk.usage.prompt_tokens
                    generated_tokens_count = chunk.usage.completion_tokens
                else:
                    input_tokens_count = 0
                    generated_tokens_count = 0
                s += content
                yield (s, SingleOutputMeta(input_tokens_count=input_tokens_count, generated_tokens_count=generated_tokens_count, finish_reason=chunk.choices[0].finish_reason))

    async def async_stream_chat_oai(
        self,
        conversations,
        role_mapping=None,
        model: Optional[str] = None,
        delta_mode: bool = False,
        llm_config: Dict[str, Any] = {},
    ):
        if not model:
            model = self.default_model_name

        deploy_info = self.deployments.get(model, {})
        client = deploy_info["async_client"]
        is_reasoning = deploy_info["is_reasoning"]
        messages, extra_params = self.process_messages(conversations)

        if is_reasoning:
            response = await client.chat.completions.create(
                messages=messages,
                model=model,
                stream=True,
                **extra_params,
            )
        else:
            response = await client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 4096),
                top_p=llm_config.get("top_p", 1.0),
                stream=True,
                **extra_params,
            )
        input_tokens_count = 0
        generated_tokens_count = 0
        if delta_mode:
            async for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens_count = chunk.usage.prompt_tokens
                    generated_tokens_count = chunk.usage.completion_tokens
                else:
                    input_tokens_count = 0
                    generated_tokens_count = 0

                yield (content, SingleOutputMeta(input_tokens_count=input_tokens_count, generated_tokens_count=generated_tokens_count, finish_reason=chunk.choices[0].finish_reason))
        else:
            s = ""
            asyncfor chunk in response:
                content = chunk.choices[0].delta.content or ""
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens_count = chunk.usage.prompt_tokens
                    generated_tokens_count = chunk.usage.completion_tokens
                else:
                    input_tokens_count = 0
                    generated_tokens_count = 0
                s += content
                yield (s, SingleOutputMeta(input_tokens_count=input_tokens_count, generated_tokens_count=generated_tokens_count, finish_reason=chunk.choices[0].finish_reason))


    def prompt(
        self,
        model: Optional[str] = None,
        render: Optional[str] = "jinja2",
        check_result: bool = False,
        options: Dict[str, Any] = {},
        return_origin_response: bool = False,
        marker: Optional[str] = None,
        assistant_prefix: Optional[str] = None,
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
                        conversations = conversations + [{"role": "assistant", "content": assistant_prefix}]

                    t = self.stream_chat_oai(
                        conversations=conversations,
                        model=model,
                        **temp_options,
                    )
                    if return_origin_response:
                        return t
                    return (item[0] for item in t)

                if issubclass(signature.return_annotation, pydantic.BaseModel):
                    response_class = signature.return_annotation
                    conversations = [{"role": "user", "content": prompt_str}]
                    if assistant_prefix:
                        conversations = conversations + [{"role": "assistant", "content": assistant_prefix}]
                    t = self.chat_oai(
                        model=model,
                        conversations=conversations,
                        response_class=response_class,
                        impl_func_params=input_dict,
                        **options,
                    )
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
                        conversations = conversations + [{"role": "assistant", "content": assistant_prefix}]                        
                    t = self.chat_oai(
                        model=model,
                        conversations=conversations,
                        **options,
                    )
                    if return_origin_response:
                        return t
                    return t[0].output
                else:
                    raise Exception(
                        f"{func.__name__} should return a pydantic model or string"
                    )

            return wrapper

        return _impl             
