import ray
import os
import time
import types
import copy
import asyncio
from typing import Any, Any, Dict, List, Tuple, Generator, Optional, Union
from byzerllm.utils.metrics import Metric
from byzerllm.utils import (
    VLLMStreamServer,
    StreamOutputs,
    SingleOutput,
    SingleOutputMeta,
    compute_max_new_tokens,
    tokenize_stopping_sequences,
)
from byzerllm.utils.tokenizer import (
    get_real_tokenizer,
    get_local_tokenizer,
    validate_args_engine_use_ray,
)
from byzerllm.utils.ray_utils import get_actor_info


try:
    from vllm.engine.async_llm_engine import (
        AsyncLLMEngine,
        AsyncEngineArgs,
        _AsyncLLMEngine,
    )
    from vllm import SamplingParams
    from vllm.utils import random_uuid
    from vllm.lora.request import LoRARequest
except ImportError:
    print(
        "vllm is not installed, if you want to use vllm backend,please install it by `pip install vllm`",
        flush=True,
    )
    pass
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    GenerationConfig,
)
from byzerllm.utils.types import StopSequencesCriteria
from ray.util.client.common import ClientActorHandle


INFERENCE_NAME = "auto"
INFER_TOKEN_METRICS = Metric()


def get_bool(params: Dict[str, str], key: str, default: bool = False) -> bool:
    if key in params:
        if isinstance(params[key], bool):
            return params[key]
        else:
            return params[key] == "true" or params[key] == "True"
    return default


def get_int(params: Dict[str, str], key: str, default: int = 0) -> int:
    if key in params:
        return int(params[key])
    return default


def get_float(params: Dict[str, str], key: str, default: float = 0.0) -> float:
    if key in params:
        return float(params[key])
    return default


def get_str(params: Dict[str, str], key: str, default: str = "") -> str:
    if key in params:
        return params[key]
    return default


async def async_get_meta(model):
    model: AsyncLLMEngine = model
    config = await model.get_model_config()
    tokenizer = model.local_tokenizer
    final_tokenizer = get_real_tokenizer(tokenizer)

    support_chat_template = (
        hasattr(final_tokenizer, "apply_chat_template")
        and hasattr(final_tokenizer, "chat_template")
        and final_tokenizer.chat_template is not None
    )

    meta = {
        "model_deploy_type": "proprietary",
        "backend": "ray/vllm",
        "support_stream": True,
        "support_chat_template": support_chat_template,
        "max_model_len": config.max_model_len,
        "architectures": getattr(config.hf_config, "architectures", []),
    }

    if not isinstance(model.engine, _AsyncLLMEngine):
        try:
            state = get_actor_info(model.engine)
            meta["engien_state"] = state.state
            meta["engine_actor_id"] = state.actor_id
            #  meta["engine_placement_group_id"] = model.placement_group.id.hex()
        except Exception as e:
            print(f"get engine state error:{e}", flush=True)

    return [meta]


async def async_vllm_chat(
    model,
    tokenizer,
    ins: str,
    his: List[Tuple[str, str]] = [],
    max_length: int = 4096,
    top_p: float = 0.95,
    temperature: float = 0.1,
    **kwargs,
):

    if "abort" in kwargs and "request_id" in kwargs:
        abort = get_bool(kwargs, "abort", False)
        request_id = kwargs["request_id"]
        if abort:
            await model.abort(request_id)
        return [
            (
                "",
                {
                    "metadata": {
                        "request_id": request_id,
                    }
                },
            )
        ]

    stream = get_bool(kwargs, "stream", False)
    request_id = kwargs["request_id"] if "request_id" in kwargs else random_uuid()
    n: int = 1
    best_of: Optional[int] = get_int(kwargs, "best_of", None)
    presence_penalty: float = float(kwargs.get("presence_penalty", 0.0))
    frequency_penalty: float = float(kwargs.get("frequency_penalty", 0.0))
    top_k: int = int(kwargs.get("top_k", -1))
    use_beam_search: bool = get_bool(kwargs, "use_beam_search", False)
    stop: Union[None, str, List[str]] = kwargs["stop"] if "stop" in kwargs else None
    ignore_eos: bool = get_bool(kwargs, "ignore_eos", False)
    max_tokens: int = max_length
    logprobs: Optional[int] = get_int(kwargs, "logprobs", None)

    adapter_name_or_path: Optional[str] = kwargs.get("adapter_name_or_path", None)
    lora_name: Optional[str] = get_str(kwargs, "lora_name", "default")
    lora_int_id: Optional[str] = get_int(kwargs, "lora_int_id", 1)

    lora_request = None
    if adapter_name_or_path is not None:
        lora_request = LoRARequest(
            lora_name=lora_name, lora_int_id=lora_int_id, lora_local_path=adapter_name_or_path
        )

    # repetition_penalty: float = float(kwargs.get("repetition_penalty",1.1))

    other_params = {}
    if "early_stopping" in kwargs:
        other_params["early_stopping"] = bool(kwargs["early_stopping"])

    if "repetition_penalty" in kwargs:
        other_params["repetition_penalty"] = float(kwargs["repetition_penalty"])

    if "stop_token_ids" in kwargs:
        stop_token_ids = kwargs["stop_token_ids"]
        if isinstance(stop_token_ids, str):
            stop_token_ids = [int(i) for i in stop_token_ids.split(",")]
        else:
            stop_token_ids = kwargs["stop_token_ids"]
        other_params["stop_token_ids"] = stop_token_ids

    sampling_params = SamplingParams(
        temperature=temperature,
        n=n,
        best_of=best_of,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        top_k=top_k,
        use_beam_search=use_beam_search,
        stop=stop,
        ignore_eos=ignore_eos,
        logprobs=logprobs,
        top_p=top_p,
        max_tokens=max_tokens,
        **other_params,
    )

    current_time_milliseconds = int(time.time() * 1000)

    if stream:
        server = ray.get_actor("VLLM_STREAM_SERVER")

        async def writer():
            results_generator = model.generate(
                ins, sampling_params, request_id, lora_request=lora_request
            )
            async for request_output in results_generator:
                v = StreamOutputs(
                    outputs=[
                        SingleOutput(
                            text=item.text,
                            metadata=SingleOutputMeta(
                                input_tokens_count=len(request_output.prompt_token_ids),
                                generated_tokens_count=len(item.token_ids),
                            ),
                        )
                        for item in request_output.outputs
                    ]
                )
                await server.add_item.remote(request_output.request_id, v)
            # mark the request is done
            await server.mark_done.remote(request_output.request_id)

        asyncio.create_task(writer())
        await server.add_item.remote(request_id, "RUNNING")
        return [
            (
                "",
                {
                    "metadata": {
                        "request_id": request_id,
                        "stream_server": "VLLM_STREAM_SERVER",
                    }
                },
            )
        ]

    results_generator = model.generate(ins, sampling_params, request_id,lora_request=lora_request)
    final_output = None
    first_token_time = current_time_milliseconds
    async for request_output in results_generator:
        if (
            first_token_time == current_time_milliseconds
            and request_output.outputs
            and len(request_output.outputs[0].token_ids) > 0
        ):
            first_token_time = int(time.time() * 1000)
        final_output = request_output
    assert final_output is not None

    text_outputs = [output for output in final_output.outputs]
    generated_text = text_outputs[0].text
    prob = text_outputs[0].cumulative_logprob

    current_time_milliseconds2 = int(time.time() * 1000)

    input_tokens_count = len(final_output.prompt_token_ids)
    generated_tokens_count = len(text_outputs[0].token_ids)
    time_cost = current_time_milliseconds2 - current_time_milliseconds
    print(
        f"cost: {time_cost}ms first_token:{first_token_time-current_time_milliseconds}ms speed: {float(generated_tokens_count)/time_cost*1000} tokens/s total_tokens_count:{input_tokens_count + generated_tokens_count} request_id:{final_output.request_id}  input_tokens_count:{input_tokens_count} generated_tokens_count:{generated_tokens_count}",
        flush=True,
    )

    INFER_TOKEN_METRICS.inc(
        f"infer_{INFERENCE_NAME}_input_tokens_num",
        input_tokens_count,
        tags={"request_id": final_output.request_id},
    )
    INFER_TOKEN_METRICS.inc(
        f"infer_{INFERENCE_NAME}_output_tokens_num",
        generated_tokens_count,
        tags={"request_id": final_output.request_id},
    )
    INFER_TOKEN_METRICS.push()

    return [
        (
            generated_text,
            {
                "metadata": {
                    "request_id": final_output.request_id,
                    "input_tokens_count": input_tokens_count,
                    "generated_tokens_count": generated_tokens_count,
                    "time_cost": time_cost,
                    "first_token_time": first_token_time - current_time_milliseconds,
                    "speed": float(generated_tokens_count) / time_cost * 1000,
                    "prob": prob,
                }
            },
        )
    ]


def from_vllm_backend(
    model_path, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}
):
    infer_mode = sys_conf.get("infer_backend", "transformers")
    num_gpus = int(sys_conf.get("num_gpus", 1))
    print(f"infer_mode:{infer_mode} tensor_parallel_size: {num_gpus}")
    global INFERENCE_NAME
    INFERENCE_NAME = infer_params.get("udfName", "auto")

    try:
        ray.get_actor("VLLM_STREAM_SERVER")
    except ValueError:
        try:
            ray.remote(VLLMStreamServer).options(
                name="VLLM_STREAM_SERVER", lifetime="detached", max_concurrency=1000
            ).remote()
        except Exception as e:
            print(f"create VLLM_STREAM_SERVER error:{e}", flush=True)

    worker_use_ray: bool = get_bool(infer_params, "backend.worker_use_ray", True)

    engine_use_ray: bool = validate_args_engine_use_ray()
    if "backend.engine_use_ray" in infer_params:
        engine_use_ray = get_bool(infer_params, "backend.engine_use_ray", False)

    tensor_parallel_size: int = num_gpus
    gpu_memory_utilization: float = float(
        infer_params.get("backend.gpu_memory_utilization", 0.90)
    )
    disable_log_stats: bool = get_bool(infer_params, "backend.disable_log_stats", False)

    other_params = {}

    def parse_value(v):
        if not isinstance(v, str):
            return v
        v = v.strip()
        if v.lower() in ['true', 'false']:
            return v.lower() == 'true'
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    for k, v in infer_params.items():
        if k.startswith("backend.") and k not in [
            "backend.worker_use_ray",
            "backend.engine_use_ray",
            "backend.gpu_memory_utilization",
            "backend.disable_log_stats",
        ]:
            new_k = k[len("backend."):]
            other_params[new_k] = parse_value(v)

    engine_args = AsyncEngineArgs(
        engine_use_ray=engine_use_ray,
        model=model_path,
        worker_use_ray=worker_use_ray,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_log_stats=disable_log_stats,
        **other_params,
    )
    llm = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = get_local_tokenizer(llm, engine_args)
    llm.local_tokenizer = tokenizer
    llm.async_stream_chat = types.MethodType(async_vllm_chat, llm)
    llm.async_get_meta = types.MethodType(async_get_meta, llm)
    return (llm, tokenizer)
