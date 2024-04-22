import os
import time
import ray
import asyncio
from typing import List, Dict, Any

from llama_cpp import Llama
from byzerllm.utils import (
    VLLMStreamServer,
    StreamOutputs,
    SingleOutput,
    SingleOutputMeta,
    compute_max_new_tokens,
)
from byzerllm.utils.types import StopSequencesCriteria


class LlamaCppBackend:
    @staticmethod
    def init_model(model_dir, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}):
        model = Llama(model_path=os.path.join(model_dir, "model.bin"))
        return model, None

    @staticmethod
    def generate(model, tokenizer, ins: str, his: List[Dict[str, str]] = [],
                 max_length: int = 4090, top_p: float = 0.95, temperature: float = 0.1, **kwargs):
        
        if model.get_meta()[0]["message_format"]:
            config = copy.deepcopy(model.generation_config)
            config.max_length = max_length
            config.temperature = temperature
            config.top_p = top_p

            if "max_new_tokens" in kwargs:
                config.max_new_tokens = int(kwargs["max_new_tokens"])

            conversations = his + [{"content": ins, "role": "user"}]
            start_time = time.monotonic()
            response = model.chat(conversations, generation_config=config)
            time_taken = time.monotonic() - start_time

            generated_tokens_count = tokenizer(response, return_token_type_ids=False, return_tensors="pt")["input_ids"].shape[1]
            print(f"chat took {time_taken} s to complete. tokens/s:{float(generated_tokens_count) / time_taken}", flush=True)

            return [(response, {"metadata": {
                "request_id": "",
                "input_tokens_count": -1,
                "generated_tokens_count": generated_tokens_count,
                "time_cost": time_taken,
                "first_token_time": -1.0,
                "speed": float(generated_tokens_count) / time_taken * 1000,
                "prob": -1.0
            }})]

        timeout_s = float(kwargs.get("timeout_s", 60 * 5))
        skip_check_min_length = int(kwargs.get("stopping_sequences_skip_check_min_length", 0))
        
        input_tokens = tokenizer(ins, return_token_type_ids=False)["input_ids"]
        max_new_tokens = compute_max_new_tokens(input_tokens, min(max_length, model.n_ctx))
        
        other_params = {}

        if "early_stopping" in kwargs:
            other_params["early_stopping"] = bool(kwargs["early_stopping"])

        if "repetition_penalty" in kwargs:
            other_params["repetition_penalty"] = float(kwargs["repetition_penalty"])

        stream = kwargs.get("stream", False)        

        current_time_milliseconds = int(time.time() * 1000)

        request_id = kwargs["request_id"] if "request_id" in kwargs else os.urandom(16).hex()

        if stream:
            server = ray.get_actor("VLLM_STREAM_SERVER")

            async def writer():
                results_generator = model(
                    ins,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    stop=kwargs["stop"] if "stop" in kwargs else [],
                    stream=True,
                    **other_params
                )

                first_token_time = current_time_milliseconds
                input_tokens_count = len(input_tokens)
                generated_tokens_count = 0

                async for text in results_generator:
                    generated_tokens_count += 1
                    if first_token_time == current_time_milliseconds:
                        first_token_time = int(time.time() * 1000)

                    v = StreamOutputs(outputs=[
                        SingleOutput(
                            text=text,
                            metadata=SingleOutputMeta(
                                input_tokens_count=input_tokens_count,
                                generated_tokens_count=generated_tokens_count,
                            )
                        )
                    ])
                    await server.add_item.remote(request_id, v)

                time_cost = int(time.time() * 1000) - current_time_milliseconds
                await server.add_item.remote(request_id, StreamOutputs(outputs=[
                    SingleOutput(
                        text="",
                        metadata=SingleOutputMeta(
                            input_tokens_count=input_tokens_count,
                            generated_tokens_count=generated_tokens_count,
                            time_cost=time_cost,
                            first_token_time=first_token_time - current_time_milliseconds,
                            speed=float(generated_tokens_count) / time_cost * 1000,
                            prob=-1.0,
                        )
                    )
                ]))
                await server.mark_done.remote(request_id)

            asyncio.create_task(writer())
            await server.add_item.remote(request_id, "RUNNING")
            return [("", {"metadata": {"request_id": request_id, "stream_server": "VLLM_STREAM_SERVER"}})]

        start_time = time.monotonic()
        result = model(
            ins,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            stop=kwargs["stop"] if "stop" in kwargs else [],
            **other_params
        )
        time_taken = time.monotonic() - start_time
        generated_text = result['choices'][0]['text']
        generated_tokens_count = len(tokenizer(generated_text)["input_ids"])

        return [(generated_text, {"metadata": {
            "request_id": request_id,
            "input_tokens_count": len(input_tokens),
            "generated_tokens_count": generated_tokens_count,
            "time_cost": time_taken,
            "first_token_time": -1.0,
            "speed": float(generated_tokens_count) / time_taken * 1000,
            "prob": -1.0
        }})]
    
    @staticmethod
    def get_meta(self):
        return [{
            "model_deploy_type": "proprietary",
            "backend": "llama.cpp",
            "max_model_len": self.n_ctx,
            "architectures": ["LlamaCpp"],
            "message_format": True
        }]