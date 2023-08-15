import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from byzerllm.stable_diffusion.api.models.diffusion import ImageGenerationOptions
from byzerllm.stable_diffusion.utils import img2b64


def save_image_base64(img: Image.Image, opts: ImageGenerationOptions):
    metadata = PngInfo()
    metadata.add_text("parameters", opts.json())
    prompt = opts.prompt
    img64 = img2b64(img)
    return (prompt, img64)


def get_image_parameter(img: Image.Image):
    text = img.text
    parameters = text.pop("parameters", None)
    try:
        text.update(json.loads(parameters))
    except:
        text.update({"parameters": parameters})
    return text
