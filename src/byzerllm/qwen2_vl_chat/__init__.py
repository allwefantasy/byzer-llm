import json
import os
import base64
import uuid
import tempfile

from transformers import Qwen2VLForConditionalGeneration, GenerationConfig
from qwen_vl_utils import process_vision_info
from typing import Dict


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
                max_length: int = 4096,
                top_p: float = 0.95,
                temperature: float = 0.1, **kwargs):
    image = kwargs["image"]
    image_b = base64.b64decode(image)

    temp_image_dir = kwargs["temp_image_dir"] if "temp_image_dir" in kwargs else os.path.join(tempfile.gettempdir(),
                                                                                              "byzerllm", "visualglm",
                                                                                              "images")

    if not os.path.exists(temp_image_dir):
        os.makedirs(temp_image_dir)

    image_file = kwargs["input_image_path"] if "input_image_path" in kwargs else os.path.join(temp_image_dir,
                                                                                              f"{str(uuid.uuid4())}.jpg")

    with open(image_file, "wb") as f:
        f.write(image_b)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
                },
                {"type": "text", "text": ins},
            ],
        }
    ]

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

    output = json.dumps({"response": output_text}, ensure_ascii=False)
    return [(output, {"metadata": {}})]


def init_model(model_dir, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}):
    pretrained_model_dir = os.path.join(model_dir, "pretrained_model")
    adaptor_model_dir = model_dir
    is_adaptor_model = os.path.exists(pretrained_model_dir)

    if not is_adaptor_model:
        pretrained_model_dir = model_dir

    tokenizer = AutoProcessor.from_pretrained(pretrained_model_dir)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        pretrained_model_dir,
        attn_implementation="flash_attention_2",
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
