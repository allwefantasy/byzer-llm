import glob
import json
import os
import re
from datetime import datetime

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from byzerllm.stable_diffusion.api.models.diffusion import ImageGenerationOptions
from byzerllm.stable_diffusion.config import (
    OUTPUT_DIR_IMG2IMG,
    OUTPUT_DIR_TEXT2IMG,
    OUTPUT_NAME_IMG2IMG,
    OUTPUT_NAME_TEXT2IMG,
)
from byzerllm.stable_diffusion.utils import img2b64


def get_category(opts: ImageGenerationOptions):
    return "img2img" if opts.image is not None else "txt2img"


def replace_invalid_chars(filepath, replace_with="_"):
    invalid_chars = '[\\/:*?"<>|]'

    replace_with = replace_with

    return re.sub(invalid_chars, replace_with, filepath)


def save_image_base64(img: Image.Image, opts: ImageGenerationOptions):
    metadata = PngInfo()
    metadata.add_text("parameters", opts.json())
    prompt = opts.prompt
    img64 = img2b64(img)
    return (prompt, img64)


def save_image(img: Image.Image, opts: ImageGenerationOptions):
    metadata = PngInfo()
    metadata.add_text("parameters", opts.json())
    dir = OUTPUT_DIR_TEXT2IMG if get_category(opts) == "txt2img" else OUTPUT_DIR_IMG2IMG
    basename = (
        OUTPUT_NAME_TEXT2IMG if get_category(opts) == "txt2img" else OUTPUT_NAME_IMG2IMG
    )
    filename = (
        basename.format(
            seed=opts.seed,
            index=len(os.listdir(dir)) + 1 if os.path.exists(dir) else 0,
            prompt=opts.prompt[:20].replace(" ", "_"),
            date=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        .replace("\n", "_")
        .replace("\r", "_")
        .replace("\t", "_")
    )
    filename = replace_invalid_chars(filename)
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, filename)
    img.save(filepath, pnginfo=metadata)
    return filepath

    dir = OUTPUT_DIR_TEXT2IMG if category == "txt2img" else OUTPUT_DIR_IMG2IMG
    return os.path.join(dir, filename)


def get_image(category: str, filename: str):
    return Image.open(get_image_filepath(category, filename))


def get_image_parameter(img: Image.Image):
    text = img.text
    parameters = text.pop("parameters", None)
    try:
        text.update(json.loads(parameters))
    except:
        text.update({"parameters": parameters})
    return text


def get_all_image_files(category: str):
    dir = OUTPUT_DIR_TEXT2IMG if category == "txt2img" else OUTPUT_DIR_IMG2IMG
    files = glob.glob(os.path.join(dir, "*"))
    files = sorted(
        [f.replace(os.sep, "/") for f in files if os.path.isfile(f)],
        key=os.path.getmtime,
    )
    files.reverse()
    return [os.path.relpath(f, dir) for f in files]
