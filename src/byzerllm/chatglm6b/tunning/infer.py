from .utils import load_pretrained
from .utils.config import (
    ModelArguments,
    DataTrainingArguments,
    FinetuningArguments
)
from transformers import HfArgumentParser
from typing import List,Any
import os

def init_model(model_path:str):
    parser = HfArgumentParser((ModelArguments,DataTrainingArguments,FinetuningArguments))
    args = [ "--model_name_or_path",model_path ]
    pretrained_model = os.path.join(model_path,"pretrained_model")
    if os.path.exists(pretrained_model):
        args = [ "--checkpoint_dir",model_path,
                 "--model_name_or_path",pretrained_model ]

    model_args, training_args, finetuning_args = parser.parse_args_into_dataclasses(args=args)
    model, tokenizer = load_pretrained(model_args, training_args, finetuning_args, is_trainable=False)
    model = model.cuda()
    model.eval()
    return (model,tokenizer)

def predict(query:str,model,tokenizer,max_length=512, top_p=0.95,temperature=0.1):    
    response = model.stream_chat(tokenizer, query, [], max_length=max_length, top_p=top_p,temperature=temperature)
    last = ""
    for t,_ in response:                                               
        last=t        
    return last