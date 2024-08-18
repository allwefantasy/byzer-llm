import time
from typing import List, Tuple, Dict, Any, Union
import base64
import json
import librosa
import io
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

def get_meta(self):
    return [
        {
            "model_deploy_type": "proprietary",
            "backend": "transformers",
            "message_format": True
        }
    ]

def process_input(ins: Union[str, List[Dict[str, Any]], Dict[str, Any]]):
    if isinstance(ins, list) or isinstance(ins, dict):
        return ins

    content = []
    try:
        ins_json = json.loads(ins)
    except:
        return ins

    if isinstance(ins_json, dict):
        return ins_json

    for item in ins_json:
        if ("audio" in item or "audio_url" in item) and "type" not in item:
            audio_data = item.get("audio", item.get("audio_url", ""))
            if not audio_data.startswith("data:"):
                audio_data = "data:audio/wav;base64," + audio_data

            # other_fields = {k: v for k, v in item.items() if k not in ["audio", "text", "audio_url"]}
            content.append(
                {
                    "audio_url": audio_data,
                    "type": "audio",
                }
            )

        if "text" in item and "type" not in item:
            text_data = item["text"]
            content.append({"text": text_data, "type": "text"})

        if "type" in item and item["type"] in ["text", "audio"]:
            content.append(item)

    if not content:
        return ins

    return content

def stream_chat(
    self,
    tokenizer,
    ins: str,
    his: List[Dict[str, str]] = [],
    max_length: int = 4096,
    top_p: float = 0.95,
    temperature: float = 0.1,
    **kwargs,
):
    messages = [
        {"role": message["role"], "content": process_input(message["content"])}
        for message in his
    ] + [{"role": "user", "content": process_input(ins)}]

    text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in messages:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audio_data = ele["audio_url"].split(",")[1]
                    audio_bytes = base64.b64decode(audio_data)
                    audio = librosa.load(io.BytesIO(audio_bytes), sr=self.processor.feature_extractor.sampling_rate)[0]
                    audios.append(audio)
        
    inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to(self.device)

    start_time = time.monotonic()
    generate_ids = self.generate(**inputs, max_length=max_length)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    time_cost = time.monotonic() - start_time

    return [
        (
            response,
            {
                "metadata": {
                    "request_id": "",
                    "input_tokens_count": inputs.input_ids.size(1),
                    "generated_tokens_count": generate_ids.size(1),
                    "time_cost": time_cost,
                    "first_token_time": 0,
                    "speed": float(generate_ids.size(1)) / time_cost if time_cost > 0 else 0,
                }
            },
        )
    ]

def init_model(model_dir, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}):
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_dir, device_map="auto")
    model.processor = processor
    import types
    model.stream_chat = types.MethodType(stream_chat, model)
    model.get_meta = types.MethodType(get_meta, model)    
    return (model, None)