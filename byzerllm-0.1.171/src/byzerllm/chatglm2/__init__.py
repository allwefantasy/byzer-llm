from transformers import AutoTokenizer, AutoModel
import transformers
import torch
import os
from typing import Any,Any,Dict, List,Tuple,Generator


from pyjava.api.mlsql import DataServer
from .. import BlockRow
from .. import parse_params

  
def get_meta(self): 
    config = self.config   
    return [{
        "model_deploy_type": "proprietary",
        "backend":"transformers",
        "max_model_len":getattr(config, "model_max_length", -1),
        "architectures":getattr(config, "architectures", [])
    }]

def stream_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = tokenizer(ins, return_token_type_ids=False,return_tensors="pt").to(device)
    response = self.generate(
        input_ids=tokens["input_ids"],
        max_new_tokens=max_length,
        repetition_penalty=1.05,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(response[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)
    return [(answer,"")]


def init_model(model_dir,infer_params:Dict[str,str]={},sys_conf:Dict[str,str]={}): 
    pretrained_model_dir = os.path.join(model_dir,"pretrained_model")
    adaptor_model_dir = model_dir
    is_adaptor_model = os.path.exists(pretrained_model_dir)
    
    if not is_adaptor_model:        
        pretrained_model_dir = model_dir

    params = parse_params(infer_params,"infer")
    load_in_4bit = params.get("load_in_4bit",False)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,trust_remote_code=True)    
    model = AutoModel.from_pretrained(pretrained_model_dir,trust_remote_code=True ).half().cuda()
    if is_adaptor_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adaptor_model_dir)
        
    model.eval()       
    import types
    # model.stream_chat = types.MethodType(stream_chat, model)  
    model.get_meta = types.MethodType(get_meta, model)     
    return (model,tokenizer)


def sft_train(data_refs:List[DataServer],
              train_params:Dict[str,str],
              conf: Dict[str, str])->Generator[BlockRow,Any,Any]:
    from ..utils.sft import sft_train as common_sft_train
    return common_sft_train(data_refs,train_params,conf) 



