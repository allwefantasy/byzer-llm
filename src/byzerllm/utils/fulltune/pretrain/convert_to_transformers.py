from deepspeed.utils import zero_to_fp32
from typing import NoReturn
import torch
import os
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import AutoTokenizer,AutoModelForCausalLM

def convert_deepspeed_checkpoint_to_transformers(model_dir:str,                                                 
                                                 checkpoint_dir:str, 
                                                 output_dir:str,tag=None) -> NoReturn:
    """
    Convert the model to transformers format.
    model: you can create it like this, BaiChuanForCausalLM(BaiChuanConfig()) 
    tokenizer_dir: here we can read and then save to the output_dir
    checkpoint_dir: the deepspeed checkpoint
    tag: epoch directory. for example 'Epoch-1'
    """    

    temp_dir = os.path.join(output_dir,"temp")   

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_pytorch_model_file = os.path.join(output_dir,"temp",WEIGHTS_NAME)
    zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, temp_pytorch_model_file, tag=tag)

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.load_state_dict(torch.load(temp_pytorch_model_file))
    model.save_pretrained(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(output_dir)


