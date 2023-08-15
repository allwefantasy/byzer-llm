import gc
import os
from typing import *

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import convert_from_ckpt
from transformers import CLIPTextModel
from byzerllm.stable_diffusion.config import stableDiffusionConfig

from byzerllm.stable_diffusion.shared import (
    hf_diffusers_cache_dir,
    hf_transformers_cache_dir,
)


def convert_checkpoint_to_pipe(model_id: str):
    if stableDiffusionConfig.get_checkpoint():
        if os.path.exists(model_id) and os.path.exists(model_id):
            return convert_from_ckpt.download_from_original_stable_diffusion_ckpt(
                model_id,
                from_safetensors=model_id.endswith(".safetensors"),
                load_safety_checker=False,
            )
    else:
        raise Exception(
            f"No {model_id} found.In checkpoint mode, model_dir must be a file.Please set the checkpoint path in the model_dir config."
        )


def load_unet(
    model_id: str, device: Optional[torch.device] = None
) -> UNet2DConditionModel:
    temporary_pipe = convert_checkpoint_to_pipe(model_id)
    if temporary_pipe is not None:
        unet = temporary_pipe.unet
        del temporary_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", cache_dir=hf_diffusers_cache_dir()
        )
    unet = unet.to(device=device)
    return unet


def load_text_encoder(model_id: str, device: Optional[torch.device] = None):
    temporary_pipe = convert_checkpoint_to_pipe(model_id)
    if temporary_pipe is not None:
        text_encoder = temporary_pipe.text_encoder
        del temporary_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", cache_dir=hf_transformers_cache_dir()
        )
    text_encoder = text_encoder.to(device=device)
    return text_encoder


def load_vae_decoder(model_id: str, device: Optional[torch.device] = None):
    temporary_pipe = convert_checkpoint_to_pipe(model_id)
    if temporary_pipe is not None:
        vae = temporary_pipe.vae
        del temporary_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", cache_dir=hf_diffusers_cache_dir()
        )

    vae.forward = vae.decode
    vae = vae.to(device=device)
    return vae


def load_vae_encoder(model_id: str, device: Optional[torch.device] = None):
    temporary_pipe = convert_checkpoint_to_pipe(model_id)
    if temporary_pipe is not None:
        vae = temporary_pipe.vae
        del temporary_pipe
        gc.collect()
        torch.cuda.empty_cache()
    else:
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", cache_dir=hf_diffusers_cache_dir()
        )

    def encoder_forward(x):
        return vae.encode(x).latent_dist.sample()

    vae.forward = encoder_forward
    vae = vae.to(device=device)
    return vae
