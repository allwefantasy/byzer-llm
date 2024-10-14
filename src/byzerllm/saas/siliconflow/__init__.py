import time
from typing import List, Tuple, Dict, Any, Union
import httpx
import requests
import base64
import io
import json
import requests
import ray
from byzerllm.utils.types import (
    BlockVLLMStreamServer,
    StreamOutputs,
    SingleOutput,
    SingleOutputMeta,
    BlockBinaryStreamServer,
)
from byzerllm.utils.langutil import asyncfy_with_semaphore
import threading
import asyncio
import traceback
import uuid
import tempfile
from loguru import logger


class CustomSaasAPI:

    def __init__(self, infer_params: Dict[str, str]) -> None:

        self.api_key = infer_params["saas.api_key"]
        self.model = infer_params.get("saas.model", "black-forest-labs/FLUX.1-dev")

        self.meta = {
            "model_deploy_type": "saas",
            "backend": "saas",
            "support_stream": False,
            "model_name": self.model,
        }

        try:
            ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
        except ValueError:
            try:
                ray.remote(BlockVLLMStreamServer).options(
                    name="BLOCK_VLLM_STREAM_SERVER",
                    lifetime="detached",
                    max_concurrency=1000,
                ).remote()
            except Exception as e:
                pass
        try:
            ray.get_actor("BlockBinaryStreamServer")
        except ValueError:
            try:
                ray.remote(BlockBinaryStreamServer).options(
                    name="BlockBinaryStreamServer",
                    lifetime="detached",
                    max_concurrency=1000,
                ).remote()
            except Exception as e:
                pass

    def get_meta(self):
        return [self.meta]

    def process_input(self, ins: Union[str, List[Dict[str, Any]], Dict[str, Any]]):
        if isinstance(ins, list) or isinstance(ins, dict):
            return ins
        try:
            return json.loads(ins)
        except:
            return ins

    async def async_text_to_image(
        self, stream: bool, input: str, size: str, quality: str, n: int, **kwargs
    ):
        if stream:
            raise Exception("Stream not supported for text to image")

        start_time = time.monotonic()
        url = "https://api.siliconflow.cn/v1/image/generations"

        payload = {
            "model": self.model,
            "prompt": input,
            "image_size": size,
            "num_inference_steps": kwargs.get("num_inference_steps", 20),
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = await asyncfy_with_semaphore(
            lambda: requests.post(url, json=payload, headers=headers)
        )()
        response_data = response.json()

        time_cost = time.monotonic() - start_time
        
        if "images" not in response_data:
            print(response_data)
            raise Exception(response_data)
        
        image_url = response_data["images"][0]["url"]
        image_response = await asyncfy_with_semaphore(lambda: requests.get(image_url))()
        image_binary = image_response.content
        # base64_image = f"data:image/png;base64,{base64.b64encode(image_binary).decode('utf-8')}"
        base64_image = base64.b64encode(image_binary).decode("utf-8")

        return [
            (
                base64_image,
                {
                    "metadata": {
                        "request_id": "",
                        "input_tokens_count": 0,
                        "generated_tokens_count": 0,
                        "time_cost": time_cost,
                        "first_token_time": 0,
                        "speed": 0,
                    }
                },
            )
        ]

    async def async_stream_chat(
        self,
        tokenizer,
        ins: str,
        his: List[Dict[str, Any]] = [],
        max_length: int = 4096,
        top_p: float = 0.7,
        temperature: float = 0.9,
        **kwargs,
    ):
        logger.info(f"[{self.model}] request accepted: {ins[-50:]}....")

        messages = [
            {"role": message["role"], "content": self.process_input(message["content"])}
            for message in his
        ] + [{"role": "user", "content": self.process_input(ins)}]

        last_message = messages[-1]["content"]

        if isinstance(last_message, dict) and "input" in last_message:
            input = last_message["input"]
            size = last_message.get("size", "1024x1024")
            quality = last_message.get("quality", "standard")
            n = last_message.get("n", 1)
            # Get other parameters excluding input, size, quality, and n
            other_params = {
                k: v
                for k, v in last_message.items()
                if k not in ["input", "size", "quality", "n"]
            }

            return await self.async_text_to_image(
                stream=False,
                input=input,
                size=size,
                quality=quality,
                n=n,
                **other_params,
            )
        raise Exception("Invalid input")
