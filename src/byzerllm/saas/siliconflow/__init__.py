import time
from typing import List, Tuple, Dict, Any, Union
import httpx
import requests
import base64
import io
import json
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
            "num_inference_steps": kwargs.get("num_inference_steps", 20)
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post(url, json=payload, headers=headers)
        response_data = response.json()
        
        time_cost = time.monotonic() - start_time
        base64_image = response_data["images"][0]["url"]
        
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
