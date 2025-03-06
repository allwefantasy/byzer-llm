import whisper
from typing import Dict, List, Tuple
from byzerllm.utils.types import StopSequencesCriteria

from typing import Dict, Any, List, Union
from pyjava.storage import streaming_tar as STar
import json
import base64
import tempfile
import time


def get_meta(self):
    return [
        {
            "model_deploy_type": "proprietary",
            "backend": "transformers",
            "message_format": True
        }
    ]


def process_input(ins: Union[str, List[Dict[str, Any]], Dict[str, Any]]):
    # print(ins) 
    if isinstance(ins, list) or isinstance(ins, dict):
        return ins

    content = []
    try:
        ins_json = json.loads(ins)
    except:
        return ins

    ## speech
    if isinstance(ins_json, dict):
        return ins_json

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
                k: v for k, v in item.items() if k not in ["image", "text", "image_url"]
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


def stream_chat(
    self,
    tokenizer,
    ins: str,
    his: List[Dict[str, str]] = [],
    max_length: int = 4090,
    top_p: float = 0.95,
    temperature: float = 0.1,
    **kwargs,
):
    messages = [
        {"role": message["role"], "content": process_input(message["content"])}
        for message in his
    ] + [{"role": "user", "content": process_input(ins)}]

    audio_input = messages[-1]["content"]

    audio_data = audio_input.get("audio", "")
    audio = audio_input["audio"]
    response_format = audio_input.get("response_format", "json")
    timestamp_granularities = audio_input.get(
        "timestamp_granularities", ["word", "segment"]
    )
    data_prefix = "data:audio/"
    base64_prefix = ";base64,"
    if not audio.startswith(data_prefix):
        raise ValueError("Invalid audio data format")

    format_end = audio.index(base64_prefix)
    audio_format = audio[len(data_prefix) : format_end]
    base64_data = audio[format_end + len(base64_prefix) :]

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
        word_timestamps = "word" in timestamp_granularities
        
        result = self.transcribe(temp_audio_file_path, word_timestamps=word_timestamps)        
            
        time_cost = time.monotonic() - start_time
        return [
            (
                json.dumps(result, ensure_ascii=False),
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


def init_model(
    model_dir, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}
):
    model = whisper.load_model("large", download_root=model_dir)
    import types

    model.stream_chat = types.MethodType(stream_chat, model)
    model.get_meta = types.MethodType(get_meta, model)
    return (model, None)
