import torch, io, time, base64
from typing import Dict, List


def get_meta(self):
    config = self.config
    return [{
        "model_deploy_type": "proprietary",
        "backend": "transformers",
        "message_format": True,
        "support_stream": False,
        "embedding_mode": False
    }]

def stream_chat(self,
                tokenizer,
                ins: str,
                his: List[Dict[str, str]] = [],
                size: str = "1024x1024", 
                num_inference_steps: int = 50,
                guidance_scale: float = 5.0,
                seed: int = 66,
                **kwargs):
    # Parse size string into height and width
    try:
        if not size:
            size = "1024x1024"
        width, height = map(int, size.split('x'))
    except (ValueError, AttributeError, TypeError):
        # Default to 1024x1024 if size parsing fails
        width, height = 1024, 1024

    # Validate dimensions are multiples of 8
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"Height ({height}) and width ({width}) must be multiples of 8")

    start_time = time.time()
    print(f"【Byzer --> Kolors Generating image with parameters: Prompt: {ins}, Size: {size} ({width}x{height}), Steps: {num_inference_steps}, Guidance Scale: {guidance_scale}, Seed: {seed}")
    # Generate image
    image = self(
        prompt=ins,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        generator=torch.Generator(self.device).manual_seed(seed)
    ).images[0]

    # Convert PIL image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Encode to base64
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

    time_cost = time.time() - start_time

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


def init_model(model_dir, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}):
    from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
    from kolors.models.modeling_chatglm import ChatGLMModel
    from kolors.models.tokenization_chatglm import ChatGLMTokenizer
    from diffusers import UNet2DConditionModel, AutoencoderKL
    from diffusers import EulerDiscreteScheduler

    text_encoder = ChatGLMModel.from_pretrained(
        f'{model_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{model_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{model_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{model_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{model_dir}/unet", revision=None).half()

    model = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=False).to("cuda")

    import types
    model.stream_chat = types.MethodType(stream_chat, model)
    model.get_meta = types.MethodType(get_meta, model)
    return (model, tokenizer)
