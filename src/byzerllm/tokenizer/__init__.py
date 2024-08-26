import json
from typing import Dict, List, Union, Any
from tokenizers import Tokenizer


def get_meta(self):
    return [
        {
            "model_deploy_type": "proprietary",
            "backend": "tokenizers",
            "message_format": True,
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

    encoded = self.encode(json.dumps(messages, ensure_ascii=False))
    return [
        (
            f"{len(encoded.tokens)}",
            {
                "metadata": {
                    "request_id": "",
                    "input_tokens_count": len(encoded.tokens),
                    "generated_tokens_count": 0,
                    "time_cost": 0,
                    "first_token_time": 0,
                    "speed": 0,
                }
            },
        )
    ]


def init_model(
    model_dir, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}
):

    model = Tokenizer.from_file(model_dir)
    import types

    model.stream_chat = types.MethodType(stream_chat, model)
    model.get_meta = types.MethodType(get_meta, model)

    return (model, None)


# Helper functions from web_ui.py
