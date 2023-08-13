import time
import traceback
from typing import Dict, List

from byzerllm.stable_diffusion.api.models.diffusion import (
    HiresfixOptions,
    ImageGenerationOptions,
    MultidiffusionOptions,
)
from byzerllm.stable_diffusion.api.models.tensorrt import BuildEngineOptions
from byzerllm.stable_diffusion.model import DiffusersModel
from byzerllm.stable_diffusion.acceleration.tensorrt.engine import EngineBuilder

# model_name = "runwayml/stable-diffusion-v1-5"


def stream_chat(
    self,
    tokenizer,
    prompt: str,
    negative_prompt: str,
    his: List[Dict[str, str]] = [],
    max_length: int = 4090,
    top_p: float = 0.95,
    temperature: float = 0.1,
    **kwargs,
):
    # images = self(ins).images
    images = generate_image_by_diffusers(
        self, prompt=prompt, negative_prompt=negative_prompt
    )
    return images


def init_model(
    model_dir, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}
):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # all params from infer_params can set in method from_pretrained
    # model = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    # model.to(device)
    # import types
    # model.stream_chat = types.MethodType(stream_chat, model)
    # return (model, None)
    model = DiffusersModel(model_dir)
    model.activate()
    import types

    model.stream_chat = types.MethodType(stream_chat, model)
    return (model, None)


# sampler_name SCHEDULERS.keys()
# samping_steps min=1,max=100
# batch_size min=1,max=50
# batch_count min=1,max=50
# cfg_scale min=1, max=20,
# seed default=-1
# width min=64,max=2048
# height min=64,max=2048
# scale_slider min=1,max=4
def generate_image_by_diffusers(
    model,
    prompt,
    negative_prompt,
    sampler_name="euler_a",
    sampling_steps=25,
    batch_size=1,
    batch_count=1,
    cfg_scale=7.5,
    seed=-1,
    width=768,
    height=768,
    enable_hires=False,
    enable_multidiff=False,
    upscaler_mode="bilinear",
    scale_slider=1.5,
    views_batch_size=4,
    window_size=64,
    stride=16,
    init_image=None,
    strength=0.5,
):
    hiresfix = HiresfixOptions(
        enable=enable_hires, mode=upscaler_mode, scale=scale_slider
    )

    multidiffusion = MultidiffusionOptions(
        enable=enable_multidiff,
        views_batch_size=views_batch_size,
        window_size=window_size,
        stride=stride,
    )

    opts = ImageGenerationOptions(
        prompt=prompt,
        negative_prompt=negative_prompt,
        batch_size=batch_size,
        batch_count=batch_count,
        scheduler_id=sampler_name,
        num_inference_steps=sampling_steps,
        guidance_scale=cfg_scale,
        height=height,
        width=width,
        strength=strength,
        seed=seed,
        image=init_image,
        hiresfix=hiresfix,
        multidiffusion=multidiffusion,
    )

    count = 0

    if opts.hiresfix.enable:
        inference_steps = opts.num_inference_steps + int(
            opts.num_inference_steps * opts.strength
        )
    else:
        inference_steps = opts.num_inference_steps

    start = time.perf_counter()

    try:
        for data in model(opts, {}):
            if type(data) == tuple:
                step, preview = data
                progress = step / (opts.batch_count * inference_steps)
                previews = []
                for images, opts in preview:
                    previews.extend(images)

                if len(previews) == count:
                    pass
                else:
                    count = len(previews)

                print(f"Progress: {progress * 100:.2f}%, Step: {step}")
            else:
                image = data

        end = time.perf_counter()

        results = []
        for images, opts in image:
            results.extend(images)

        print(f"Finished in {end -start:0.4f} seconds")
        yield results
    except Exception as e:
        traceback.print_exc()
        yield []


# opt_image_height min=1 max=2048
# opt_image_width min=1 max=2048
# min_latent_resolution min=1 max=2048
# max_latent_resolution min=1 max=2048
# onnx_opset min=7 max=18
def generate_image_by_tensorrt(
    model,
    max_batch_size=1,
    opt_image_height=512,
    opt_image_width=512,
    min_latent_resolution=256,
    max_latent_resolution=1024,
    build_enable_refit=False,
    build_static_batch=False,
    build_dynamic_shape=True,
    build_all_tactics=False,
    build_preview_features=True,
    onnx_opset=17,
    force_engine_build=False,
    force_onnx_export=False,
    force_onnx_optimize=False,
    full_acceleration=False,
):
    print("Building Engine...")
    model.teardown()
    opts = BuildEngineOptions(
        max_batch_size=max_batch_size,
        opt_image_height=opt_image_height,
        opt_image_width=opt_image_width,
        min_latent_resolution=min_latent_resolution,
        max_latent_resolution=max_latent_resolution,
        build_enable_refit=build_enable_refit,
        build_static_batch=build_static_batch,
        build_dynamic_shape=build_dynamic_shape,
        build_all_tactics=build_all_tactics,
        build_preview_features=build_preview_features,
        onnx_opset=onnx_opset,
        force_engine_build=force_engine_build,
        force_onnx_export=force_onnx_export,
        force_onnx_optimize=force_onnx_optimize,
        full_acceleration=full_acceleration,
    )
    builder = EngineBuilder(model, opts)
    builder.build()
    model.activate()
    print("Engine built finish...")
