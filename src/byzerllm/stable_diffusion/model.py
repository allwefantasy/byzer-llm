from queue import Queue
import gc
import random
from typing import Dict, List, Literal
from concurrent.futures import ThreadPoolExecutor

import torch
from byzerllm.stable_diffusion import utils
from byzerllm.stable_diffusion.api.models.diffusion import ImageGenerationOptions
from byzerllm.stable_diffusion.config import stableDiffusionConfig

from byzerllm.stable_diffusion.diffusion.piplines.diffusers import DiffusersPipeline
from byzerllm.stable_diffusion.images import save_image_base64
from byzerllm.stable_diffusion.lib.diffusers.scheduler import (
    SCHEDULERS,
    parser_schedulers_config,
)
from byzerllm.stable_diffusion.shared import get_device, hf_diffusers_cache_dir
from packaging.version import Version


ModelMode = Literal["diffusers"]
PrecisionMap = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}


class DiffusersModel:
    def __init__(
        self,
        model_id: str,
        checkpoint: bool = False,
        variant: str = "fp16",
        mode: str = "diffusers",
    ):
        self.model_id: str = model_id
        self.mode: ModelMode = mode
        self.activated: bool = False
        self.pipe = None
        self.variant = variant
        self.checkpoint = checkpoint

    def available_modes(self):
        modes = ["diffusers"]

        return modes

    def activate(self):
        if self.activated:
            return
        device = get_device()

        precision = stableDiffusionConfig.get_precision() or "fp32"
        torch_dtype = PrecisionMap[precision]

        if self.mode == "diffusers":
            self.pipe = DiffusersPipeline.from_pretrained(
                self.model_id,
                use_auth_token=stableDiffusionConfig.get_hf_token(),
                torch_dtype=torch_dtype,
                variant=self.variant,
                cache_dir=hf_diffusers_cache_dir(),
                checkpoint=self.checkpoint,
            ).to(device=device)

            if Version(torch.__version__) < Version("2"):
                self.pipe.enable_attention_slicing()

            if (
                utils.is_installed("xformers")
                and stableDiffusionConfig.get_xformers()
                and device.type == "cuda"
            ):
                self.pipe.enable_xformers_memory_efficient_attention()
        self.activated = True

    def teardown(self):
        if not self.activated:
            return
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        self.activated = False

    def change_mode(self, mode: ModelMode):
        if mode == self.mode:
            return
        self.teardown()
        self.mode = mode
        self.activate()

    def swap_scheduler(self, scheduler_id: str):
        if not self.activated:
            raise RuntimeError("Model not activated")
        self.pipe.scheduler = SCHEDULERS[scheduler_id].from_config(
            self.pipe.scheduler.config, **parser_schedulers_config(scheduler_id)
        )

    def __call__(self, opts: ImageGenerationOptions, plugin_data: Dict[str, List] = {}):
        if not self.activated:
            raise RuntimeError("Model not activated")

        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        self.swap_scheduler(opts.scheduler_id)

        queue = Queue()
        done = object()
        total_steps = 0

        results = []

        def callback(*args, **kwargs):
            nonlocal total_steps
            total_steps += 1
            queue.put((total_steps, results))

        def on_done(feature):
            queue.put(done)

        for i in range(opts.batch_count):
            manual_seed = int(opts.seed + i)

            generator = torch.Generator(device=self.pipe.device).manual_seed(
                manual_seed
            )

            with ThreadPoolExecutor() as executer:
                feature = executer.submit(
                    self.pipe,
                    opts=opts,
                    generator=generator,
                    callback=callback,
                    plugin_data=plugin_data,
                )
                feature.add_done_callback(on_done)

                while True:
                    item = queue.get()
                    if item is done:
                        break
                    yield item

                images = feature.result().images

            results.append(
                (
                    [save_image_base64(img, opts) for img in images],
                    ImageGenerationOptions.parse_obj(
                        {"seed": manual_seed, **opts.dict()}
                    ),
                )
            )

        yield results
