from typing import *


class StableDiffusionConfig:
    hf_token = ""
    temp_dir = "stable_diffusion_models"
    xformers = True
    model_dir = None
    precision = "fp16"
    checkpoint = False

    def set_checkpoint(self, checkpoint):
        self.checkpoint = checkpoint

    def set_precision(self, precision):
        self.precision = precision

    def set_model_dir(self, model_dir):
        self.model_dir = model_dir

    def set_temp_dir(self, temp_dir):
        self.temp_dir = temp_dir

    def set_hf_token(self, hf_token):
        self.hf_token = hf_token

    def set_xformers(self, xformers):
        self.xformers = xformers

    def get_checkpoint(self):
        return self.checkpoint

    def get_precision(self):
        return self.precision

    def get_model_dir(self):
        if self.model_dir is None:
            raise Exception("Model dir not set")
        return self.model_dir

    def get_temp_dir(self):
        return self.temp_dir

    def get_hf_token(self):
        return self.hf_token

    def get_xformers(self):
        return self.xformers


stableDiffusionConfig = StableDiffusionConfig()
