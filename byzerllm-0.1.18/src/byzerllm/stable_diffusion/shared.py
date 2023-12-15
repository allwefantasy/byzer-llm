import os

import torch

from byzerllm.stable_diffusion.config import stableDiffusionConfig


def hf_diffusers_cache_dir():
    cache_dir = os.path.join(stableDiffusionConfig.get_temp_dir(), "diffusers")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def hf_transformers_cache_dir():
    cache_dir = os.path.join(stableDiffusionConfig.get_temp_dir(), "transformers")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
