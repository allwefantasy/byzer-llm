import os
import re
from glob import glob
from typing import *

import torch
import byzerllm.stable_diffusion.diffusion.networks.lyco as lyco
from byzerllm.stable_diffusion.logger import set_logger
from byzerllm.stable_diffusion.api.events import event_handler
from byzerllm.stable_diffusion.api.events.generation import LoadResourceEvent
from byzerllm.stable_diffusion.api.models.diffusion import ImageGenerationOptions
from byzerllm.stable_diffusion.config import stableDiffusionConfig


logger = set_logger(__name__)

latest_networks: List[Tuple[str, torch.nn.Module]] = []


def get_networks_from_prompt(prompt: str) -> list:
    networks = []
    for network in re.findall(
        r"<(.*?)>", string=prompt, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
    ):
        network = network.split(":")
        if len(network) < 3:
            continue
        networks.append(network)

    replaced = re.sub(r"<(.+)\:(.+)\:(.+)>", "", prompt)

    return networks, replaced


def find_network_filepath(network: str, subfolder: str) -> str:
    models_dir = os.path.join(stableDiffusionConfig.get_temp_dir(), subfolder)

    for file in glob(os.path.join(models_dir, "**", "*"), recursive=True):
        for ext in [".pt", ".safetensors"]:
            if os.path.basename(file) == f"{network}{ext}":
                return file


def restore_networks(*modules: torch.nn.Module):
    global latest_networks

    for _, _, network in latest_networks[::-1]:
        network.restore(*modules)


@event_handler()
def load_network_modules(e: LoadResourceEvent):
    global latest_networks

    opts: ImageGenerationOptions = e.pipe.session.opts

    positive_networks, opts.prompt = get_networks_from_prompt(opts.prompt)

    changed = False

    if len(positive_networks) == len(latest_networks):
        for next, prev in zip(positive_networks, latest_networks):
            if (
                next[0] != prev[0]
                or next[1] != prev[1]
                or float(next[2]) != prev[2].multiplier
            ):
                changed = True
                break
    else:
        changed = True

    if not changed:
        return

    restore_networks(e.pipe.unet, e.pipe.text_encoder)
    latest_networks.clear()

    if len(positive_networks) == 0:
        return

    for module_type, basename, multiplier in positive_networks:
        multiplier = float(multiplier)
        if module_type == "lora":
            filepath = find_network_filepath(basename, "lora")
            network_module = lyco.LycorisNetwork
        elif module_type == "lyco":
            filepath = find_network_filepath(basename, "lycoris")
            network_module = lyco
        else:
            continue

        if filepath is None:
            logger.warn(f"network {basename} is not found")
            continue

        network, weights_sd = network_module.create_network_from_weights(
            multiplier,
            filepath,
            e.pipe.vae,
            e.pipe.text_encoder,
            e.pipe.unet,
        )
        info = network.load_state_dict(weights_sd, False)
        network.set_multiplier(multiplier)
        logger.info(f"weights are loaded: {info}")
        if hasattr(network, "merge_to"):
            network.merge_to()
        else:
            network.apply_to()
        network = network.to(device=e.pipe.device, dtype=e.pipe.dtype)
        latest_networks.append((module_type, basename, network))

    logger.info(
        f'loaded networks: {", ".join([basename for _, basename, _ in latest_networks])}'
    )


def init():
    pass
