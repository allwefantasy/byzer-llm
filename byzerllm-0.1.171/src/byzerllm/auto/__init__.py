import ray
import os
import time
import types
import copy
import asyncio
from typing import Any,Any,Dict, List,Tuple,Generator,Optional,Union
from pyjava.api.mlsql import DataServer
from byzerllm.utils.metrics import Metric
from byzerllm import BlockRow
from byzerllm.utils import (VLLMStreamServer,
                            StreamOutputs,
                            SingleOutput,
                            SingleOutputMeta,
                            compute_max_new_tokens,
                            tokenize_stopping_sequences,
                            ) 
from byzerllm.utils.tokenizer import get_real_tokenizer,get_local_tokenizer,validate_args_engine_use_ray                        
from byzerllm.utils.ray_utils import get_actor_info

try:
    from byzerllm.auto.backend_llama_cpp import LlamaCppBackend
except ImportError:
    print("python_llama_cpp is not installed, if you want to use llama_cpp backend,please install it by `pip install python_llama_cpp`",flush=True)
    pass

try:
    from vllm.engine.async_llm_engine import AsyncLLMEngine,AsyncEngineArgs,_AsyncLLMEngine    
    from vllm import  SamplingParams
    from vllm.utils import random_uuid    
except ImportError:
    print("vllm is not installed, if you want to use vllm backend,please install it by `pip install vllm`",flush=True)
    pass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig,StoppingCriteriaList,GenerationConfig
from byzerllm.utils.types import StopSequencesCriteria
from ray.util.client.common import ClientActorHandle

INFERENCE_NAME = "auto"
INFER_TOKEN_METRICS = Metric()

def get_bool(params:Dict[str,str],key:str,default:bool=False)->bool:
    if key in params:
        if isinstance(params[key],bool):
            return params[key]
        else:            
            return params[key] == "true" or params[key] == "True"
    return default

def get_int(params:Dict[str,str],key:str,default:int=0)->int:
    if key in params:
        return int(params[key])
    return default

def get_float(params:Dict[str,str],key:str,default:float=0.0)->float:
    if key in params:
        return float(params[key])
    return default

def get_str(params:Dict[str,str],key:str,default:str="")->str:
    if key in params:
        return params[key]
    return default    


def stream_chat(self,tokenizer,ins:str, his:List[Dict[str,str]]=[],  
        max_length:int=4090, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
 
    if self.get_meta()[0]["message_format"]:
        config = copy.deepcopy(self.generation_config)
        config.max_length = max_length
        config.temperature = temperature
        config.top_p = top_p        
        
        if "max_new_tokens" in kwargs:
            config.max_new_tokens = int(kwargs["max_new_tokens"])
        
        conversations = his + [{"content":ins,"role":"user"}]
        start_time = time.monotonic()
        response = self.chat(tokenizer, messages=conversations,generation_config=config)
        time_taken = time.monotonic() - start_time

        generated_tokens_count = tokenizer(response, return_token_type_ids=False,return_tensors="pt")["input_ids"].shape[1]
        print(f"chat took {time_taken} s to complete. tokens/s:{float(generated_tokens_count)/time_taken}",flush=True)
        
        return [(response,{"metadata":{
            "request_id":"",
            "input_tokens_count": -1,
            "generated_tokens_count":generated_tokens_count,
            "time_cost":time_taken,
            "first_token_time": -1.0,
            "speed":float(generated_tokens_count)/time_taken*1000,
            "prob": -1.0
        }})] 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    timeout_s = float(kwargs.get("timeout_s",60*5)) 
    skip_check_min_length = int(kwargs.get("stopping_sequences_skip_check_min_length",0))       
    
    tokens = tokenizer(ins, return_token_type_ids=False,return_tensors="pt").to(device)

    stopping_criteria = None


    if "stopping_sequences" in kwargs:        
        stopping_sequences = [torch.tensor(word).to(device) for word in tokenize_stopping_sequences(tokenizer,kwargs["stopping_sequences"].split(","))]    
        input_length = tokens["input_ids"].shape[1]
        stopping_criteria=StoppingCriteriaList([StopSequencesCriteria(
            tokenizer=tokenizer,
            stops=stopping_sequences,
            input_start=input_length,
            skip_check_min_length=skip_check_min_length
            )])

    config = self.config

    max_new_tokens = compute_max_new_tokens(tokens, min(max_length, getattr(config, "model_max_length", max_length))) 

    other_params = {}  
    if "early_stopping" in kwargs:
        other_params["early_stopping"] = bool(kwargs["early_stopping"])

    if "repetition_penalty" in kwargs:
        other_params["repetition_penalty"] = float(kwargs["repetition_penalty"])

    if self.generation_config and self.generation_config.eos_token_id:
        other_params["eos_token_id"] = self.generation_config.eos_token_id

    if self.generation_config and self.generation_config.pad_token_id:
        other_params["pad_token_id"] = self.generation_config.pad_token_id
    
    if self.generation_config and self.generation_config.bos_token_id:
        other_params["bos_token_id"] = self.generation_config.bos_token_id
    
    start_time = time.monotonic()        
    response = self.generate(
        input_ids=tokens["input_ids"],
        max_new_tokens= max_new_tokens,        
        temperature=temperature,
        top_p=top_p,        
        max_time=timeout_s,
        stopping_criteria=stopping_criteria,
        **other_params
    )    
    time_taken = time.monotonic() - start_time    
    new_tokens = response[0][tokens["input_ids"].shape[1]:]
    print(f"generate took {time_taken} s to complete. tokens/s:{len(new_tokens)/time_taken}",flush=True)
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return [(answer,{"metadata":{
            "request_id":"",
            "input_tokens_count": tokens["input_ids"].shape[1],
            "generated_tokens_count":len(new_tokens),
            "time_cost":time_taken,
            "first_token_time": -1.0,
            "speed":float(generated_tokens_count)/time_taken*1000,
            "prob": -1.0
        }})] 


def init_model(model_dir,infer_params:Dict[str,str]={},sys_conf:Dict[str,str]={}): 
    infer_mode = sys_conf.get("infer_backend","transformers")
    quatization = infer_params.get("quatization","false") == "true"  

    if infer_mode == "llama_cpp":       
        model = LlamaCppBackend(model_path=model_dir,infer_params=infer_params,sys_conf=sys_conf)
        return (model,None)

    if infer_mode == "ray/vllm":        
        from byzerllm.auto.backend_vllm import from_vllm_backend
        llm,tokenizer = from_vllm_backend(model_path=model_dir,infer_params=infer_params,sys_conf=sys_conf)
        return (llm,tokenizer)  

    if  infer_mode == "ray/deepspeed":
        from .backend_ds import DeepSpeedInference,ParallelConfig        
        num_gpus = int(sys_conf.get("num_gpus",1))
        model = DeepSpeedInference(ParallelConfig(num_workers=num_gpus,model_dir=model_dir))    
        return (model,None)                     

    pretrained_model_dir = os.path.join(model_dir,"pretrained_model")
    adaptor_model_dir = model_dir
    is_adaptor_model = os.path.exists(pretrained_model_dir)
    
    if not is_adaptor_model:        
        pretrained_model_dir = model_dir

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,trust_remote_code=True)    

    quatization = infer_params.get("backend.quantization",infer_params.get("quatization", "false"))

    if quatization in ["4", "8", "true",True,4,8]:
        print(f"enable [{quatization}] quatization.", flush=True)
        load_in_8bit = quatization == "8" or quatization == 8
        # default using int4
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if load_in_8bit:
            llm_int8_threshold = infer_params.get("llm_int8_threshold", 6.0)
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=llm_int8_threshold,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False,
            )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_dir,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir,trust_remote_code=True,
                                                device_map='auto',                                                
                                                torch_dtype=torch.bfloat16                                                
                                                )
    if is_adaptor_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adaptor_model_dir)
    
    model.generation_config = GenerationConfig.from_pretrained(pretrained_model_dir)
    
    if "use_cache" in infer_params or "backend.use_cache" in infer_params:
        use_cache = infer_params.get("backend.use_cache",infer_params.get("use_cache", "false"))
        if isinstance(use_cache,bool):
            model.generation_config.use_cache = use_cache
        else:
            model.generation_config.use_cache = use_cache == "true"    
    
    model.eval()  
    if quatization:
        model = torch.compile(model)
           
    has_chat = hasattr(model,"chat")
    
    if "message_format" in infer_params or "backend.message_format" in infer_params:
        message_format = infer_params.get("backend.message_format",infer_params.get("message_format", "false"))
        if isinstance(message_format,bool):
            has_chat = message_format
        else:
            has_chat = message_format == "true"

    extra_meta = {}
    if has_chat:
        extra_meta["message_format"] = True

    def get_meta(self): 
        config = self.config           
        return [{
            "model_deploy_type": "proprietary",
            "backend":"transformers",
            "max_model_len":getattr(config, "model_max_length", -1),
            "architectures":getattr(config, "architectures", []),
            **extra_meta
        }]    

    model.stream_chat = types.MethodType(stream_chat, model)
    model.get_meta = types.MethodType(get_meta, model)     
    return (model,tokenizer)


def sft_train(data_refs:List[DataServer],
              train_params:Dict[str,str],
              conf: Dict[str, str])->Generator[BlockRow,Any,Any]:
    from ..utils.sft import sft_train as common_sft_train
    return common_sft_train(data_refs,train_params,conf) 

