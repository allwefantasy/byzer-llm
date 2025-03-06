import json
from typing import List, Dict, Any,Union
from loguru import logger
import threading
import asyncio
import uuid
import ray
import aiofiles
from byzerllm.utils.types import (
    BlockVLLMStreamServer,
    StreamOutputs,
    SingleOutput,
    SingleOutputMeta,
    BlockBinaryStreamServer,
)


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.log_file = infer_params.get("saas.log_file", "filelogger.json")
        self.model = infer_params.get("saas.model", "gpt-3.5-turbo-1106")        
        self.meta = {
            "model_deploy_type": "saas",
            "backend": "filelogger",
            "support_stream": True,
            "model_name": "filelogger",
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

    def process_input(self, ins: Union[str, List[Dict[str, Any]], Dict[str, Any]]):

        if isinstance(ins, list) or isinstance(ins, dict):
            return ins

        content = []
        try:
            ins_json = json.loads(ins)
        except:
            return ins

        ## 如果是字典，应该是非chat格式需求，比如语音转文字等
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
                ## "data:image/jpeg;base64,"
                if not image_data.startswith("data:"):
                    image_data = "data:image/jpeg;base64," + image_data

                ## get the other fields except image/text/image_url
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

            ## for format like this: {"type": "text", "text": "What’s in this image?"},
            ## {"type": "image_url", "image_url": {"url":"","detail":"high"}}
            ## this is the standard format, just return it
            if "type" in item and item["type"] == "text":
                content.append(item)

            if "type" in item and item["type"] == "image_url":
                content.append(item)

        if not content:
            return ins

        return content

    def get_meta(self):
        return [self.meta]

    async def chat_oai(self, conversations: List[Dict[str, Any]], **kwargs):
        try:
            async with asyncio.Lock():
                async with aiofiles.open(self.log_file, "a") as f:
                    await f.write(json.dumps(conversations, ensure_ascii=False) + "\n")
            return [("Messages logged successfully", {})]
        except Exception as e:
            logger.error(f"Error logging messages: {e}")
            return [(f"Error logging messages: {e}", {})]

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

        messages = [
            {"role": message["role"], "content": self.process_input(message["content"])}
            for message in his
        ] + [{"role": "user", "content": self.process_input(ins)}]

        stream = kwargs.get("stream", False)
        if not stream:
            return await self.chat_oai(messages, **kwargs)

        server = ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
        request_id = str(uuid.uuid4())

        async def writer():
            try:
                message = "Messages logged successfully"
                for i in range(len(message)):
                    chunk = message[: i + 1]
                    await server.add_item.remote(
                        request_id,
                        StreamOutputs(
                            outputs=[
                                SingleOutput(
                                    text=chunk,
                                    metadata=SingleOutputMeta(
                                        input_tokens_count=0,
                                        generated_tokens_count=i + 1,
                                    ),
                                )
                            ]
                        ),
                    )
                await server.mark_done.remote(request_id)
            except Exception as e:
                logger.error(f"Error in stream writing: {e}")
                await server.mark_done.remote(request_id)

        asyncio.create_task(writer())

        await server.add_item.remote(request_id, "RUNNING")
        return [
            (
                "",
                {
                    "metadata": {
                        "request_id": request_id,
                        "stream_server": "BLOCK_VLLM_STREAM_SERVER",
                    }
                },
            )
        ]
