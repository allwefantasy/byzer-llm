from pydantic import BaseModel
from byzerllm.utils.nontext import Image,Audio


class Bool(BaseModel):
    value: bool


class Int(BaseModel):
    value: int


class Float(BaseModel):
    value: float


class ImagePath(BaseModel):
    value: str

    def __str__(self):
        return Image.load_image_from_path(self.value)


class AudioPath(BaseModel):
    value: str

    def __str__(self):
        return Audio.load_audio_from_path(self.value)
