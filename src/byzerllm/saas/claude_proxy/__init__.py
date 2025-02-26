import asyncio
import json
import threading
import time
import traceback
import ray
import requests
from typing import List, Dict, Any, Union

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
        self.model = infer_params.get("saas.model", "claude-3-5-sonnet-20241022")
        self.anthropic_version = infer_params.get("saas.anthropic_version", "2024-02-29")
        self.base_url = infer_params.get("saas.base_url", "https://api.anthropic.com")
        self.meta = {
            "model_deploy_type": "saas",
            "backend": "saas",
            "support_stream": True,
            "support_assistant_prefix": True,
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
                image_format = image_data[len(data_prefix): format_end]
                base64_data = image_data[format_end + len(base64_prefix):]

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
                image_format = image_data[len(data_prefix): format_end]
                base64_data = image_data[format_end + len(base64_prefix):]
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
            max_length: int = -1,
            top_p: float = 0.9,
            temperature: float = 0.1,
            **kwargs,
    ):
        messages = []
        system_message = ""
        for message in his:
            if message["role"] == "system":
                system_message = f"{system_message}\n\n{message['content']}"
            else:
                messages.append(
                    {
                        "role": message["role"],
                        "content": self.process_input(message["content"]),
                    }
                )
        process_message = self.process_input(ins)
        if isinstance(process_message, List):
            messages.extend(process_message)
        elif isinstance(ins, dict):
            messages.append(process_message)
        else:
            messages.append({"role": "user", "content": process_message})

        start_time = time.monotonic()

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json"
        }
        stream = kwargs.get("stream", False)

        payload = {
            "model": self.model,
            "temperature": temperature,
            "top_p": top_p,
            "messages": messages,
            "stream": stream
        }

        if max_length > 0:
            payload["max_tokens"] = max_length

        if system_message:
            payload["system"] = system_message
        print(payload)
        try:
            res_data = requests.post(self.base_url + "/v1/messages", json=payload, headers=headers, stream=stream)
            res_data.raise_for_status()
        except requests.exceptions.RequestException as e:
            traceback.print_exc()
            print(messages)
            if hasattr(e.response, 'text'):
                error_msg = e.response.text
            else:
                error_msg = str(e)
            raise Exception(f"Request failed: {error_msg}")

        if stream:
            server = ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
            request_id = [None]

            def writer():
                input_tokens = 0
                r = ""
                for line in res_data.iter_lines():
                    try:
                        if not line or not (line_str := line.decode('utf-8')).startswith('data: '):
                            continue
                        response = json.loads(line_str.lstrip('data: '))

                        response_type = response.get("type")
                        generated_tokens = 0
                        if response_type == "message_start":
                            request_id[0] = response["message"]["id"]
                            input_tokens = response["message"]["usage"]["input_tokens"]
                        elif response_type == "content_block_delta":
                            r += response["delta"]["text"]
                        elif response_type == "message_delta":
                            generated_tokens = response["usage"]["output_tokens"]
                        else:
                            continue

                        if response_type in ("content_block_delta", "message_delta"):
                            ray.get(server.add_item.remote(request_id[0],
                                                           StreamOutputs(
                                                               outputs=[SingleOutput(text=r, metadata=SingleOutputMeta(
                                                                   input_tokens_count=input_tokens,
                                                                   generated_tokens_count=generated_tokens,
                                                               ))])
                                                           ))
                    except Exception as e:
                        print(f"error {e.__str__()}")
                ray.get(server.mark_done.remote(request_id[0]))

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
            return [("", {"metadata": {"request_id": request_id[0], "stream_server": "BLOCK_VLLM_STREAM_SERVER"}})]
        else:
            time_cost = time.monotonic() - start_time

            try:
                response_json = res_data.json()
                generated_text = response_json["content"][0]["text"]
                generated_tokens_count = response_json["usage"]["output_tokens"]
                input_tokens_count = response_json["usage"]["input_tokens"]
                stop_reason = response_json.get("stop_reason")
                request_id = response_json.get("id")
            except (KeyError, json.JSONDecodeError) as e:
                raise Exception(f"Failed to parse response: {str(e)}")

            return [
                (
                    generated_text,
                    {
                        "metadata": {
                            "request_id": request_id,
                            "input_tokens_count": input_tokens_count,
                            "generated_tokens_count": generated_tokens_count,
                            "time_cost": time_cost,
                            "first_token_time": 0,
                            "speed": float(generated_tokens_count) / time_cost,
                            "stop_reason": stop_reason,
                        }
                    },
                )
            ]
