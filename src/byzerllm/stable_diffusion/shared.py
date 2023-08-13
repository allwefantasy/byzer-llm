import os

import torch

from byzerllm.stable_diffusion.config import MODEL_DIR


def hf_diffusers_cache_dir():
    cache_dir = os.path.join(MODEL_DIR, "diffusers")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def hf_transformers_cache_dir():
    cache_dir = os.path.join(MODEL_DIR, "transformers")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
