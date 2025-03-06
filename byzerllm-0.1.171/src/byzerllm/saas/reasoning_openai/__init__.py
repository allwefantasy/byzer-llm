import time
from typing import List, Tuple, Dict, Any, Union
import httpx
from openai import OpenAI
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
        self.model = infer_params.get("saas.model", "gpt-3.5-turbo-1106")

        other_params = {}

        if "saas.api_base" in infer_params:
            other_params["api_base"] = infer_params["saas.api_base"]

        if "saas.api_version" in infer_params:
            other_params["api_version"] = infer_params["saas.api_version"]

        if "saas.api_type" in infer_params:
            other_params["api_type"] = infer_params["saas.api_type"]

        if "saas.base_url" in infer_params:
            other_params["base_url"] = infer_params["saas.base_url"]

        if "saas.timeout" in infer_params:
            other_params["timeout"] = float(infer_params["saas.timeout"])

        self.max_retries = int(infer_params.get("saas.max_retries", 10))

        self.meta = {
            "model_deploy_type": "saas",
            "backend": "saas",
            "support_stream": True,
            "model_name": self.model,
            "support_assistant_prefix": True,
        }

        self.meta["embedding_mode"] = "embedding" in self.model.lower()

        self.proxies = infer_params.get("saas.proxies", None)
        self.local_address = infer_params.get("saas.local_address", "0.0.0.0")
        self.other_params = other_params

        if not self.proxies:
            self.client = OpenAI(**other_params, api_key=self.api_key)
        else:
            self.client = OpenAI(
                **other_params,
                api_key=self.api_key,
                http_client=httpx.Client(
                    proxies=self.proxies,
                    transport=httpx.HTTPTransport(
                        local_address=self.local_address),
                ),
            )

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

    # saas/proprietary
    def get_meta(self):
        return [self.meta]

    def process_input(self, ins: Union[str, List[Dict[str, Any]], Dict[str, Any]]):

        if isinstance(ins, list) or isinstance(ins, dict):
            return ins

        content = []
        try:
            ins_json = json.loads(ins)
        except:
            return ins

        # 如果是字典，应该是非chat格式需求，比如语音转文字等
        if isinstance(ins_json, dict):
            return ins_json

        if isinstance(ins_json, list):
            if ins_json and isinstance(ins_json[0], dict):
                # 根据key值判断是什么类型的输入，比如语音转文字，语音合成等
                for temp in ["input", "voice", "audio", "audio_url"]:
                    if temp in ins_json[0]:
                        return ins_json[0]

        content = []
        #     [
        #     {"type": "text", "text": "What’s in this image?"},
        #     {
        #       "type": "image_url",
        #       "image_url": {
        #         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        #         "detail": "high"
        #       },
        #     },
        #   ],
        for item in ins_json:
            # for format like this: {"image": "xxxxxx", "text": "What’s in this image?","detail":"high"}
            # or {"image": "xxxxxx"}, {"text": "What’s in this image?"}
            if ("image" in item or "image_url" in item) and "type" not in item:
                image_data = item.get("image", item.get("image_url", ""))
                # "data:image/jpeg;base64,"
                if not image_data.startswith("data:"):
                    image_data = "data:image/jpeg;base64," + image_data

                # get the other fields except image/text/image_url
                other_fields = {
                    k: v
                    for k, v in item.items()
                    if k not in ["image", "text", "image_url"]
                }
                content.append(
                    {
                        "image_url": {"url": image_data, **other_fields},
                        "type": "image_url",
                    }
                )

            if "text" in item and "type" not in item:
                text_data = item["text"]
                content.append({"text": text_data, "type": "text"})

            # for format like this: {"type": "text", "text": "What’s in this image?"},
            # {"type": "image_url", "image_url": {"url":"","detail":"high"}}
            # this is the standard format, just return it
            if "type" in item and item["type"] == "text":
                content.append(item)

            if "type" in item and item["type"] == "image_url":
                content.append(item)

        if not content:
            return ins

        return content

    async def async_embed_query(self, ins: str, **kwargs):
        resp = await asyncfy_with_semaphore(
            lambda: self.client.embeddings.create(
                input=[ins], model=self.model)
        )()
        embedding = resp.data[0].embedding
        usage = resp.usage
        return (
            embedding,
            {
                "metadata": {
                    "input_tokens_count": usage.prompt_tokens,
                    "generated_tokens_count": 0,
                }
            },
        )

    def embed_query(self, ins: str, **kwargs):
        resp = self.client.embeddings.create(input=[ins], model=self.model)
        embedding = resp.data[0].embedding
        usage = resp.usage
        return (
            embedding,
            {
                "metadata": {
                    "input_tokens_count": usage.prompt_tokens,
                    "generated_tokens_count": 0,
                }
            },
        )

    async def async_text_to_speech(
        self, stream: bool, ins: str, voice: str, chunk_size: int = None, **kwargs
    ):
        if stream:
            server = ray.get_actor("BlockBinaryStreamServer")
            request_id = [None]

            def writer():
                try:
                    request_id[0] = str(uuid.uuid4())
                    with self.client.with_streaming_response.audio.speech.create(
                        model=self.model, voice=voice, input=ins, **kwargs
                    ) as response:
                        for chunk in response.iter_bytes(chunk_size):
                            input_tokens_count = 0
                            generated_tokens_count = 0
                            ray.get(
                                server.add_item.remote(
                                    request_id[0],
                                    StreamOutputs(
                                        outputs=[
                                            SingleOutput(
                                                text=chunk,
                                                metadata=SingleOutputMeta(
                                                    input_tokens_count=input_tokens_count,
                                                    generated_tokens_count=generated_tokens_count,
                                                ),
                                            )
                                        ]
                                    ),
                                )
                            )
                except:
                    traceback.print_exc()
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
            return [
                (
                    "",
                    {
                        "metadata": {
                            "request_id": request_id[0],
                            "stream_server": "BlockBinaryStreamServer",
                        }
                    },
                )
            ]

        start_time = time.monotonic()
        with io.BytesIO() as output:
            async with asyncfy_with_semaphore(
                lambda: self.client.with_streaming_response.audio.speech.create(
                    model=self.model, voice=voice, input=ins, **kwargs
                )
            )() as response:
                for chunk in response.iter_bytes():
                    output.write(chunk)

            base64_audio = base64.b64encode(output.getvalue()).decode()
            time_cost = time.monotonic() - start_time
            return [
                (
                    base64_audio,
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

    async def async_speech_to_text(
        self,
        audio: str,
        response_format: str = "verbose_json",
        timestamp_granularities: List[str] = ["word", "segment"],
    ):
        # Extract audio format and base64 data
        data_prefix = "data:audio/"
        base64_prefix = ";base64,"
        if not audio.startswith(data_prefix):
            raise ValueError("Invalid audio data format")

        format_end = audio.index(base64_prefix)
        audio_format = audio[len(data_prefix): format_end]
        base64_data = audio[format_end + len(base64_prefix):]

        # Decode the base64 audio data
        audio_data = base64.b64decode(base64_data)

        # Create a temporary file with the correct extension
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{audio_format}"
        ) as temp_audio_file:
            temp_audio_file.write(audio_data)
            temp_audio_file_path = temp_audio_file.name

        try:
            start_time = time.monotonic()

            with open(temp_audio_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=self.model or "whisper-1",
                    file=audio_file,
                    response_format=response_format,
                    timestamp_granularities=timestamp_granularities,
                )
            time_cost = time.monotonic() - start_time

            return [
                (
                    json.dumps(transcription.to_dict(), ensure_ascii=True),
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
        finally:
            # Clean up the temporary file
            import os

            os.unlink(temp_audio_file_path)

    def image_to_text(self, ins: str, **kwargs):
        pass

    async def async_text_to_image(
        self, stream: bool, input: str, size: str, quality: str, n: int, **kwargs
    ):
        if stream:
            raise Exception("Stream not supported for text to image")
        start_time = time.monotonic()
        response = await asyncfy_with_semaphore(
            lambda: self.client.images.generate(
                model=self.model,
                prompt=input,
                size=size,
                quality=quality,
                n=1,
                response_format="b64_json",
                **kwargs,
            )
        )()
        time_cost = time.monotonic() - start_time
        base64_image = response.data[0].b64_json
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

    def text_to_text(self, ins: str, **kwargs):
        pass

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
        model = self.model

        if "model" in kwargs:
            model = kwargs["model"]

        logger.info(f"[{model}] request accepted: {ins[-50:]}....")

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

        stream = kwargs.get("stream", False)

        if "stop" in kwargs:
            extra_params["stop"] = (
                kwargs["stop"]
                if isinstance(kwargs["stop"], list)
                else json.loads(kwargs["stop"])
            )

        # content = [
        # "voice": "alloy","input": "Hello, World!",response_format: "mp3"]
        last_message = messages[-1]["content"]

        if isinstance(last_message, dict) and "voice" in last_message:
            voice = last_message["voice"]
            response_format = last_message.get("response_format", "mp3")
            chunk_size = last_message.get("chunk_size", None)
            input = last_message["input"]
            return await self.async_text_to_speech(
                stream=stream,
                ins=input,
                voice=voice,
                chunk_size=chunk_size,
                response_format=response_format,
            )

        if isinstance(last_message, dict) and "input" in last_message:
            input = last_message["input"]
            size = last_message.get("size", "1024x1024")
            quality = last_message.get("quality", "standard")
            n = last_message.get("n", 1)
            return await self.async_text_to_image(
                stream=stream, input=input, size=size, quality=quality, n=n
            )

        if isinstance(last_message, dict) and "audio" in last_message:
            audio_data = last_message.get("audio", "")
            tpe = last_message.get("type", "wav")
            if not audio_data.startswith("data:"):
                audio_data = f"data:audio/${tpe};base64," + audio_data
            return await self.async_speech_to_text(audio=audio_data)

        server = ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
        request_id = [None]

        if "extra_request_params" in extra_params:
            if "extra_body" in extra_params:
                extra_params["extra_body"] = {**extra_params["extra_request_params"],**extra_params["extra_body"]}
            else:
                extra_params["extra_body"] = extra_params["extra_request_params"]

        def writer():
            try:
                r = ""
                reasoning_r = ""
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=True,
                    stream_options={"include_usage": True},
                    max_tokens=max_length,
                    **extra_params,
                )
                # input_tokens_count = 0
                # generated_tokens_count = 0

                request_id[0] = str(uuid.uuid4())
                last_meta = None

                if hasattr(response, "error"):
                    raise Exception(response.error)

                for chunk in response:
                    if hasattr(chunk, "usage") and chunk.usage:
                        input_tokens_count = chunk.usage.prompt_tokens
                        generated_tokens_count = chunk.usage.completion_tokens
                    else:
                        input_tokens_count = 0
                        generated_tokens_count = 0

                    if not chunk.choices:
                        if last_meta:
                            last_meta = SingleOutput(
                                text=r,
                                metadata=SingleOutputMeta(
                                    input_tokens_count=input_tokens_count,
                                    generated_tokens_count=generated_tokens_count,
                                    reasoning_content=reasoning_r,
                                    finish_reason=last_meta.finish_reason,
                                ),
                            )
                            ray.get(
                                server.add_item.remote(
                                    request_id[0],
                                    StreamOutputs(outputs=[last_meta])
                                )
                            )
                        continue

                    content = chunk.choices[0].delta.content or ""
                    reasoning_content = ""
                    if hasattr(chunk.choices[0].delta, "reasoning_content"):
                        reasoning_content = chunk.choices[0].delta.reasoning_content or ""
                    r += content
                    reasoning_r += reasoning_content

                    last_meta = SingleOutput(
                        text=r,
                        metadata=SingleOutputMeta(
                            input_tokens_count=input_tokens_count,
                            generated_tokens_count=generated_tokens_count,
                            reasoning_content=reasoning_r,
                            finish_reason=chunk.choices[0].finish_reason,
                        ),
                    )
                    ray.get(
                        server.add_item.remote(
                            request_id[0],
                            StreamOutputs(
                                outputs=[
                                    last_meta
                                ]
                            ),
                        )
                    )
            except:
                traceback.print_exc()
            ray.get(server.mark_done.remote(request_id[0]))

        logger.info(json.dumps(messages, ensure_ascii=False, indent=2))
        base_url = self.other_params.get("base_url", "")
        logger.info(
            f"params: model={model}, max_length={max_length}, base_url={base_url},  stream={stream}"
        )
        logger.info(f"kwargs: {kwargs} extra_params: {extra_params}")
        if stream:
            threading.Thread(target=writer, daemon=True).start()

            time_count = 60 * 100
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
        else:
            try:
                start_time = time.monotonic()
                response = await asyncfy_with_semaphore(
                    lambda: self.client.chat.completions.create(
                        messages=messages,
                        model=model,
                        max_tokens=max_length,
                        **extra_params,
                    )
                )()

                if hasattr(response, "error"):
                    raise Exception(response.error)

                generated_text = response.choices[0].message.content
                generated_tokens_count = response.usage.completion_tokens
                reasoning_content = ""
                if hasattr(response.choices[0].message, "reasoning_content"):
                    reasoning_content = response.choices[0].message.reasoning_content or ""
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
                        "reasoning_content": reasoning_content,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                }
                logger.info(gen_meta)
                return [
                    (
                        generated_text,
                        gen_meta,
                    )
                ]
            except Exception as e:
                logger.error(f"Error: {e}")
                traceback.print_exc()
                raise e
