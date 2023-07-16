from transformers import AutoTokenizer, AutoModelForCausalLM
import ray
import torch
import os
import ray
from typing import Any,Any,Dict, List,Tuple,Generator


from pyjava.api.mlsql import DataServer
from .. import BlockRow



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


def tgi_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    model = self
    response = ray.get(model.generate_text.remote(ins))
    return [(response.generated_text,"")]

def vllm_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    from vllm import  SamplingParams
    model = self
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_length=max_length)
    outputs = model.generate([ins], sampling_params)    
    generated_text = outputs[0].outputs[0].text
    return [(generated_text,"")]


def init_model(model_dir,infer_params:Dict[str,str]={},sys_conf:Dict[str,str]={}): 
    infer_mode = infer_params.get("inferMode","transformers")
    

    if infer_mode == "tgi":
        import byzerllm.utils.inference as TGI
        return TGI.init_model(model_dir,infer_params)
    
    if infer_mode == "ray/tgi":        
        from byzerllm.utils.rayinfer import build_model_serving
        model = build_model_serving(model_dir)        
        model.stream_chat = types.MethodType(tgi_chat, model) 
        return (model,None) 

    if infer_mode == "ray/vllm":
        workerUseRay = infer_params.get("workerUseRay","false") == "true"
        print(f"infer_mode:{infer_mode} workerUseRay:{workerUseRay} tensor_parallel_size: {len(ray.get_gpu_ids())}")
        from vllm import LLM                
        llm = LLM(model=model_dir,
                  tensor_parallel_size=len(ray.get_gpu_ids()),
                  worker_use_ray=workerUseRay,  
                  trust_remote_code=True,                
                  disable_log_stats=False)
        llm.stream_chat = types.MethodType(vllm_chat, llm) 
        return (llm,None)                        

    pretrained_model_dir = os.path.join(model_dir,"pretrained_model")
    adaptor_model_dir = model_dir
    is_adaptor_model = os.path.exists(pretrained_model_dir)
    
    if not is_adaptor_model:        
        pretrained_model_dir = model_dir

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
    tokenizer.padding_side="right"
    tokenizer.pad_token_id=0
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir,trust_remote_code=True,
                                                device_map='auto',                                                
                                                torch_dtype=torch.bfloat16                                                
                                                )
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



