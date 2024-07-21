import whisper
from typing import Dict, List, Tuple
from byzerllm.utils.types import StopSequencesCriteria

from typing import Dict, Any, List, Generator
from pyjava.storage import streaming_tar as STar
import json
import base64
import tempfile
import time


def get_meta(self):
    config = self.config
    return [
        {
            "model_deploy_type": "proprietary",
            "backend": "transformers",
            "max_model_len": getattr(config, "model_max_length", -1),
            "architectures": getattr(config, "architectures", []),
        }
    ]


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
    audio_input = json.loads(ins)
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

        with open(temp_audio_file_path, "rb") as audio_file:
            result = self.transcribe(audio_file, word_timestamps=word_timestamps)
        time_cost = time.monotonic() - start_time
        return [
            (
                json.dumps(result, ensure_ascii=True),
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
    model = whisper.load_model("large", model_dir)
    import types

    model.stream_chat = types.MethodType(stream_chat, model)
    model.get_meta = types.MethodType(get_meta, model)
    return (model, None)
