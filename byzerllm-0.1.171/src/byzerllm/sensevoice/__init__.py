import json
import base64
import tempfile
import time
from typing import Dict, List, Union, Any
from funasr import AutoModel

emo_dict = {
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
}

event_dict = {
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "��",
    "<|Cry|>": "😭",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "🤧",
}

emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓",
    "<|zh|>": "",
    "<|en|>": "",
    "<|yue|>": "",
    "<|ja|>": "",
    "<|ko|>": "",
    "<|nospeech|>": "",
    "<|HAPPY|>": "😊",
    "<|SAD|>": "😔",
    "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "",
    "<|BGM|>": "🎼",
    "<|Speech|>": "",
    "<|Applause|>": "👏",
    "<|Laughter|>": "😀",
    "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢",
    "<|SURPRISED|>": "😮",
    "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "",
    "<|Sneeze|>": "🤧",
    "<|Breath|>": "",
    "<|Cough|>": "😷",
    "<|Sing|>": "",
    "<|Speech_Noise|>": "",
    "<|withitn|>": "",
    "<|woitn|>": "",
    "<|GBG|>": "",
    "<|Event_UNK|>": "",
}

lang_dict = {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷"}


def format_str(s):
    for sptk in emoji_dict:
        s = s.replace(sptk, emoji_dict[sptk])
    return s


def format_str_v2(s):
    sptk_dict = {sptk: s.count(sptk) for sptk in emoji_dict}
    s = "".join(c for c in s if c not in emoji_dict.values())
    emo = max(emo_dict, key=lambda e: sptk_dict[e])
    event = "".join(event_dict[e] for e in event_dict if sptk_dict[e] > 0)
    s = event + s + emo_dict[emo]
    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji).replace(emoji + " ", emoji)
    return s.strip()


def format_str_v3(s):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None

    def get_event(s):
        return s[0] if s[0] in event_set else None

    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        s = s.replace(lang, "<|lang|>")
    s_list = [format_str_v2(s_i).strip() for s_i in s.split("<|lang|>")]
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) is not None:
            s_list[i] = s_list[i][1:]
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) is not None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()


def get_meta(self):
    return [
        {
            "model_deploy_type": "proprietary",
            "backend": "funasr",
            "message_format": True,
        }
    ]


def process_input(ins: Union[str, List[Dict[str, Any]], Dict[str, Any]]):
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
        if "audio" in item and "type" not in item:
            audio_data = item.get("audio", "")
            if not audio_data.startswith("data:"):
                audio_data = f"data:audio/wav;base64,{audio_data}"
            content.append({"audio": audio_data, "type": "audio"})

        if "text" in item and "type" not in item:
            content.append({"text": item["text"], "type": "text"})

        if "type" in item:
            content.append(item)

    return content if content else ins


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

    audio = audio_input["audio"]
    data_prefix = "data:audio/"
    base64_prefix = ";base64,"
    if not audio.startswith(data_prefix):
        raise ValueError("Invalid audio data format")

    format_end = audio.index(base64_prefix)
    audio_format = audio[len(data_prefix) : format_end]
    base64_data = audio[format_end + len(base64_prefix) :]

    audio_data = base64.b64decode(base64_data)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{audio_format}"
    ) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file_path = temp_audio_file.name

    try:
        start_time = time.monotonic()

        result = self.generate(input=temp_audio_file_path, cache={})
        print(result)

        time_cost = time.monotonic() - start_time

        formatted_result = format_str_v3(result[0]["text"])

        return [
            (
                json.dumps({"text": formatted_result}, ensure_ascii=False),
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
        import os

        os.unlink(temp_audio_file_path)


def init_model(
    model_dir, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}
):
    model_kwargs = {
        "model": model_dir,
        "trust_remote_code": True,
    }

    if "remote_code" in infer_params:
        model_kwargs["remote_code"] = infer_params["remote_code"]

    if "vad_model" in infer_params:
        model_kwargs["vad_model"] = infer_params["vad_model"]
        vad_kwargs = {}
        for key, value in infer_params.items():
            if key.startswith("vad_kwargs."):
                param_name = key.split(".", 1)[1]
                # Attempt to convert the value to int or float if possible
                try:
                    vad_kwargs[param_name] = int(value)
                except ValueError:
                    try:
                        vad_kwargs[param_name] = float(value)
                    except ValueError:
                        vad_kwargs[param_name] = value  # Keep as string if not a number
        model_kwargs["vad_kwargs"] = vad_kwargs

    if "device" in infer_params:
        model_kwargs["device"] = infer_params["device"]

    model = AutoModel(**model_kwargs)

    import types

    model.stream_chat = types.MethodType(stream_chat, model)
    model.get_meta = types.MethodType(get_meta, model)

    return (model, None)


# Helper functions from web_ui.py
