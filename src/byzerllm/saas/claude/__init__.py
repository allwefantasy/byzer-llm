import asyncio
import json
import threading
import time
import traceback
from typing import List, Dict, Any, Union
import ray
from anthropic import Anthropic

from byzerllm.utils.types import (
    BlockVLLMStreamServer,
    StreamOutputs,
    SingleOutput,
    SingleOutputMeta,
)
from byzerllm.utils.langutil import asyncfy_with_semaphore


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key: str = infer_params["saas.api_key"]
        self.model = infer_params.get("saas.model", "claude-3-haiku-20240307")
        self.meta = {
            "model_deploy_type": "saas",
            "backend": "saas",
            "support_stream": True,
            "support_assistant_prefix": True,
        }
        other_params = {}

        if "saas.base_url" in infer_params:
            other_params["base_url"] = infer_params["saas.base_url"]

        self.client = Anthropic(api_key=self.api_key, **other_params)

        try:
            ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
        except ValueError:
            try:
                ray.remote(BlockVLLMStreamServer).options(
                    name="BLOCK_VLLM_STREAM_SERVER",
                    lifetime="detached",
                    max_concurrency=1000,
                ).remote()
            except:
                pass

    # saas/proprietary
    def get_meta(self):
        return [self.meta]

    async def async_get_meta(self):
        return await asyncfy_with_semaphore(self.get_meta)()

    def process_input(self, ins: Union[str, List[Dict[str, Any]], Dict[str, Any]]):
        if isinstance(ins, list) or isinstance(ins, dict):
            return ins

        content = []
        try:
            ins_json = json.loads(ins)
        except:
            return ins

        if isinstance(ins_json, dict):
            return ins_json

        content = []
        for item in ins_json:
            if "image" in item or "image_url" in item and "type" not in item:
                image_data = item.get("image", item.get("image_url", ""))
                if not image_data.startswith("data:"):
                    image_data = "data:image/jpeg;base64," + image_data

                data_prefix = "data:image/"
                base64_prefix = ";base64,"
                if not image_data.startswith(data_prefix):
                    raise ValueError("Invalid image data format")

                format_end = image_data.index(base64_prefix)
                image_format = image_data[len(data_prefix) : format_end]
                base64_data = image_data[format_end + len(base64_prefix) :]

                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image_format}",
                            "data": base64_data,
                        },
                    }
                )

            if "text" in item and "type" not in item:
                text_data = item["text"]
                content.append({"type": "text", "text": text_data})

            if "type" in item and item["type"] in ["text"]:
                content.append(item)
            # 兼容openai {"type": "image_url", "image_url": {"url":"","detail":"high"}}
            if "type" in item and item["type"] == "image_url":
                image_data = item["image_url"]["url"]
                data_prefix = "data:image/"
                base64_prefix = ";base64,"
                if not image_data.startswith(data_prefix):
                    raise ValueError("Invalid image data format")

                format_end = image_data.index(base64_prefix)
                image_format = image_data[len(data_prefix) : format_end]
                base64_data = image_data[format_end + len(base64_prefix) :]
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image_format}",
                            "data": base64_data,
                        },
                    }
                )

        if not content:
            return ins

        return content

    async def async_stream_chat(
        self,
        tokenizer,
        ins: str,
        his: List[dict] = [],
        max_length: int = 1024,
        top_p: float = 0.9,
        temperature: float = 0.1,
        **kwargs,
    ):
        messages = []
        system_message = ""
        for message in his:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                messages.append(
                    {
                        "role": message["role"],
                        "content": self.process_input(message["content"]),
                    }
                )

        messages.append({"role": "user", "content": self.process_input(ins)})
        
        # 支持assistant prefix
        if len(messages) > 1 and messages[-1]["role"] == "user" and messages[-2]["role"] == "user":
            messages[-1]["role"] = "assistant"

        start_time = time.monotonic()

        other_params = {}

        if system_message:
            other_params["system"] = system_message

        if "stream" in kwargs:
            other_params["stream"] = kwargs["stream"]

        stream = kwargs.get("stream", False)

        try:
            res_data = await asyncfy_with_semaphore(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    messages=messages,
                    **other_params,
                )
            )()
        except Exception as e:
            traceback.print_exc()
            print(messages)
            raise e

        if stream:
            server = ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
            request_id = [None]

            def writer():
                input_tokens = 0
                r = ""
                for response in res_data:

                    if response.type == "message_start":
                        request_id[0] = response.message.id
                        input_tokens = response.message.usage.input_tokens

                    if response.type == "content_block_delta":
                        v = response.delta.text
                        r += v
                        server.add_item.remote(
                            request_id[0],
                            StreamOutputs(
                                outputs=[
                                    SingleOutput(
                                        text=r,
                                        metadata=SingleOutputMeta(
                                            input_tokens_count=0,
                                            generated_tokens_count=0,
                                        ),
                                    )
                                ]
                            ),
                        )
                    if response.type == "message_delta":
                        server.add_item.remote(
                            request_id[0],
                            StreamOutputs(
                                outputs=[
                                    SingleOutput(
                                        text=r,
                                        metadata=SingleOutputMeta(
                                            input_tokens_count=input_tokens,
                                            generated_tokens_count=response.usage.output_tokens,
                                        ),
                                    )
                                ]
                            ),
                        )

                server.mark_done.remote(request_id[0])

            threading.Thread(target=writer, daemon=True).start()

            time_count = 10 * 100
            while request_id[0] is None and time_count > 0:
                await asyncio.sleep(0.01)
                time_count -= 1

            if request_id[0] is None:
                raise Exception("Failed to get request id")

            def write_running():
                return ray.get(server.add_item.remote(request_id[0], "RUNNING"))

            await asyncio.to_thread(write_running)
            return [
                (
                    "",
                    {
                        "metadata": {
                            "request_id": request_id[0],
                            "stream_server": "BLOCK_VLLM_STREAM_SERVER",
                        }
                    },
                )
            ]

        time_cost = time.monotonic() - start_time

        generated_text = res_data.content[0].text
        generated_tokens_count = res_data.usage.output_tokens
        input_tokens_count = res_data.usage.input_tokens

        return [
            (
                generated_text,
                {
                    "metadata": {
                        "request_id": res_data.id,
                        "input_tokens_count": input_tokens_count,
                        "generated_tokens_count": generated_tokens_count,
                        "time_cost": time_cost,
                        "first_token_time": 0,
                        "speed": float(generated_tokens_count) / time_cost,
                        "stop_reason": res_data.stop_reason,
                    }
                },
            )
        ]
