# import argparse
import os
from typing import *

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_TOKEN = ""
MODEL_DIR = "models"
PRECISION = "fp16"
XFORMERS = True
OUTPUT_DIR_TEXT2IMG = "outputs/txt2img"
OUTPUT_DIR_IMG2IMG = "outputs/img2img"
OUTPUT_NAME_TEXT2IMG = "{index}-{seed}-{prompt}.png"
OUTPUT_NAME_IMG2IMG = "{index}-{seed}-{prompt}.png"
TENSORRT = True
TENSORRT_FULL_ACCELERATION = True
