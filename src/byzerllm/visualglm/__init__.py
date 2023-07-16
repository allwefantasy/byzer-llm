from transformers import AutoTokenizer, AutoModel
import transformers
import torch
import os
import io
from typing import Any,Any,Dict, List,Tuple,Generator
import base64
import uuid
import tempfile

from pyjava.api.mlsql import DataServer
from .. import BlockRow
from .. import parse_params



def stream_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    image_b = base64.b64decode(kwargs["image"])

    temp_image_dir = os.path.join(tempfile.gettempdir(),"byzerllm","visualglm","images")
    if "temp_image_dir" in kwargs:
        temp_image_dir = kwargs["temp_image_dir"]

    if not os.path.exists(temp_image_dir):
        os.makedirs(temp_image_dir)

    image_file = os.path.join(temp_image_dir,f"{str(uuid.uuid4())}.jpg")
    with open(image_file,"wb") as f:
        f.write(image_b)
            
    response, history = self.chat(tokenizer,image_file,ins,his,max_length=max_length,top_p=top_p,temperature=temperature)
    
    os.remove(image_file)

    return [(response,"")]


def init_model(model_dir,infer_params:Dict[str,str]={},sys_conf:Dict[str,str]={}): 
    pretrained_model_dir = os.path.join(model_dir,"pretrained_model")
    adaptor_model_dir = model_dir
    is_adaptor_model = os.path.exists(pretrained_model_dir)
    
    if not is_adaptor_model:        
        pretrained_model_dir = model_dir

    params = parse_params(infer_params,"infer")
    load_in_4bit = params.get("load_in_4bit",False)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,trust_remote_code=True)    
    model = AutoModel.from_pretrained(pretrained_model_dir,trust_remote_code=True,
                                                load_in_4bit=load_in_4bit,
                                                device_map='auto'
                                                ).half().cuda()
    if is_adaptor_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adaptor_model_dir)
        
    model.eval()       
    import types
    model.stream_chat = types.MethodType(stream_chat, model)     
    return (model,tokenizer)


def sft_train(data_refs:List[DataServer],
              train_params:Dict[str,str],
              conf: Dict[str, str])->Generator[BlockRow,Any,Any]:
    from ..utils.sft import sft_train as common_sft_train
    return common_sft_train(data_refs,train_params,conf) 



