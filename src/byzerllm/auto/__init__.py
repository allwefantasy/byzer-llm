from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import ray
import torch
import os
import ray
from typing import Any,Any,Dict, List,Tuple,Generator
import types

from pyjava.api.mlsql import DataServer
from .. import BlockRow

def stream_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=1024, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = tokenizer(ins, return_token_type_ids=False,return_tensors="pt").to(device)
    response = self.generate(
        input_ids=tokens["input_ids"],
        max_new_tokens=max_length,
        repetition_penalty=1.05,
        temperature=temperature,
        attention_mask=tokens.attention_mask        
    )
    answer = tokenizer.decode(response[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)
    return [(answer,"")]


def ray_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    from aviary.backend.server.models import Prompt
    model = self
    response = ray.get(model.generate_text.remote(prompt=Prompt(
        prompt=ins,
        use_prompt_format=False
    ),request=None))
    return [(response.generated_text,"")]

def vllm_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    from vllm import  SamplingParams
    model = self
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_length)
    outputs = model.generate([ins], sampling_params)    
    generated_text = outputs[0].outputs[0].text
    return [(generated_text,"")]


def init_model(model_dir,infer_params:Dict[str,str]={},sys_conf:Dict[str,str]={}): 
    infer_mode = sys_conf.get("infer_backend","transformers")
    quatization = infer_params.get("quatization","false") == "true"
    
    if infer_mode == "transformers":
        raise Exception('''
transformers backend is not supported in pretrainedModelType: auto.
Try to use ray/tgi or ray/deepspeed or ray/vllm.

For example:

!byzerllm setup "infer_backend=ray/vllm";
''')
    

    if infer_mode == "tgi":
        import byzerllm.utils.inference as TGI
        return TGI.init_model(model_dir,infer_params)                
    
    if infer_mode in ["ray/deepspeed","ray/devicemap"]:   
        num_workers = int(sys_conf.get("num_gpus",1))   
        udfName = infer_params["udfName"]
        mode = infer_mode.split("/")[1]
        from byzerllm.utils.rayinfer import build_model_serving
        model = build_model_serving(udfName,model_dir, mode=mode, num_workers=num_workers)        
        model.stream_chat = types.MethodType(ray_chat, model) 
        return (model,None) 

    if infer_mode == "ray/vllm":
        workerUseRay = infer_params.get("workerUseRay","true") == "true"
        num_gpus = int(sys_conf.get("num_gpus",1))
        print(f"infer_mode:{infer_mode} workerUseRay:{workerUseRay} tensor_parallel_size: {num_gpus}")
        from vllm import LLM                
        llm = LLM(model=model_dir,
                  tensor_parallel_size=num_gpus,
                  worker_use_ray=workerUseRay,                   
                  trust_remote_code=True,                
                  disable_log_stats=False)
        # tokenizer = llm.get_tokenizer()
        # tokenizer.padding_side="right"
        # tokenizer.pad_token_id=0
        llm.stream_chat = types.MethodType(vllm_chat, llm) 
        return (llm,None)  

    if  infer_mode == "deepseed":
        import deepspeed
        num_gpus = int(sys_conf.get("num_gpus",1))
        tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)  
        AutoModelForCausalLM.from_pretrained(pretrained_model_dir,trust_remote_code=True,                                                                                                
                                                torch_dtype=torch.bfloat16                                                
                                                )       
        ds_engine = deepspeed.init_inference(model,
                                 mp_size=num_gpus,
                                 dtype=torch.half,
                                 replace_method="auto",
                                 replace_with_kernel_inject=True)
        model = ds_engine.module 
        model.stream_chat = types.MethodType(stream_chat, model)     
        return (model,tokenizer)                     

    pretrained_model_dir = os.path.join(model_dir,"pretrained_model")
    adaptor_model_dir = model_dir
    is_adaptor_model = os.path.exists(pretrained_model_dir)
    
    if not is_adaptor_model:        
        pretrained_model_dir = model_dir

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)    

    if quatization:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_dir,
            trust_remote_code=True,            
            device_map="auto",
            quantization_config=nf4_config,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir,trust_remote_code=True,
                                                device_map='auto',                                                
                                                torch_dtype=torch.bfloat16                                                
                                                )
    if is_adaptor_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adaptor_model_dir)

    model.eval()  
    if quatization:
        model = torch.compile(model)   

    model.stream_chat = types.MethodType(stream_chat, model)     
    return (model,tokenizer)


def sft_train(data_refs:List[DataServer],
              train_params:Dict[str,str],
              conf: Dict[str, str])->Generator[BlockRow,Any,Any]:
    from ..utils.sft import sft_train as common_sft_train
    return common_sft_train(data_refs,train_params,conf) 

