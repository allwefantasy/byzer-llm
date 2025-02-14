import json
import os
import base64
import uuid

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
from typing import Dict, List, Union, Any


def get_meta(self):
    config = self.config
    return [{
        "model_deploy_type": "proprietary",
        "backend": "transformers",
        "max_model_len": getattr(config, "model_max_length", -1),
        "architectures": getattr(config, "architectures", []),
        "message_format": True,
    }]


def stream_chat(self, tokenizer, ins: str,
                his: List[Dict[str, str]] = [],
                max_length: int = 4096,
                top_p: float = 0.95,
                temperature: float = 0.1, **kwargs):
    messages = [{"role": message["role"], "content": process_input(message["content"])} for message in his] + [
        {"role": "user", "content": process_input(ins)}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = tokenizer(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = self.generate(**inputs,
                                  max_length=max_length,
                                  temperature=temperature,
                                  top_p=top_p)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return [(output_text, {"metadata": {}})]


def process_input(ins: Union[str, List[Dict[str, Any]]]):
    if isinstance(ins, list):
        return ins

    try:
        ins_json = json.loads(ins)
    except:
        return ins

    content = []
    for item in ins_json:
        if "image" in item:
            image_data = item["image"]
            ## "data:image/jpeg;base64,"
            if image_data.startswith("data:"):
                [data_type, image] = image_data.split(";")
                [_, image_data] = image.split(",")
                [_, image_and_type] = data_type.split(":")
                image_type = image_and_type.split("/")[1]

            else:
                image_type = "jpg"
                image_data = image_data

            image_b = base64.b64decode(image_data)
            image_file = os.path.join("/tmp", f"{str(uuid.uuid4())}.{image_type}")
            with open(image_file, "wb") as f:
                f.write(image_b)
            content.append({"type": "image", "image": f"file://{image_file}"})
        if "text" in item:
            text_data = item["text"]
            content.append({"type": "text", "text": text_data})

    if not content:
        return ins
    return content


def init_model(model_dir, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}):
    pretrained_model_dir = os.path.join(model_dir, "pretrained_model")
    adaptor_model_dir = model_dir
    is_adaptor_model = os.path.exists(pretrained_model_dir)

    if not is_adaptor_model:
        pretrained_model_dir = model_dir

    tokenizer = AutoProcessor.from_pretrained(pretrained_model_dir)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_model_dir,
        # attn_implementation="flash_attention_2",
        torch_dtype="auto",
        device_map="auto"
    )
    if is_adaptor_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adaptor_model_dir)

    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(pretrained_model_dir, trust_remote_code=True)
    import types
    model.stream_chat = types.MethodType(stream_chat, model)
    model.get_meta = types.MethodType(get_meta, model)
    return (model, tokenizer)
