from deepspeed.utils import zero_to_fp32
from typing import NoReturn
import torch
import os
from transformers import WEIGHTS_NAME, CONFIG_NAME

def convert_deepspeed_checkpoint_to_transformers(checkpoint_dir:str, output_dir:str) -> NoReturn:
    """
    Convert the model to transformers format.
    """    
    zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, os.path.join(output_dir,WEIGHTS_NAME), tag=None)