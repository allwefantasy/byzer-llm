from typing import Any,Any,Dict, List,Tuple,Generator
from byzerllm.utils.inference.models import get_model
from byzerllm.utils.inference.models.flash_causal_lm import FlashCausalLMBatch
from byzerllm.utils.inference.models.types import BatchRequest,Request,StoppingCriteriaParameters,NextTokenChooserParameters
import torch
import torch.multiprocessing as mp
import os

def chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=2048, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    
    stop_sequences = []
    if "stop_sequences"  in kwargs:
        stop_sequences = kwargs("stop_words").split(",")
    
    request_id = 0

    query = FlashCausalLMBatch.from_client(BatchRequest(
        id=request_id,requests=[Request(id=request_id,inputs=ins,truncate=max_length/2,parameters=NextTokenChooserParameters(
            temperature=temperature,top_k=0,top_p=top_p,typical_p=0.0,do_sample=True,seed=0,repetition_penalty=1.0,watermark=False
        ),stopping_parameters=StoppingCriteriaParameters(
            max_new_tokens=max_length,stop_sequences=stop_sequences,ignore_eos_token=False
        ),prefill_logprobs=False)],
        size=1,max_tokens=max_length
    ),tokenizer,torch.bfloat16, self.device)

    self.warmup(query,max_length)
    while True:
        generations, batch = self.generate_token(query)
        if not batch:
            break

    return [(generations[0].generated_text.text,"")]

def _init_model(model_dir,infer_params:Dict[str,str]={}): 
    sharded = infer_params.get("model.sharded",True)
    quantize = infer_params.get("model.quantize","bitsandbytes")
    model = get_model(model_dir,
                      revision=None,
                      sharded=sharded,                      
                      quantize=quantize,
                      dtype="bfloat16",
                      trust_remote_code=True)
    model.eval()       
    import types
    model.stream_chat = types.MethodType(chat, model)     
    return (model,model.tokenizer)    

def get_available_port():
    import socket    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
    s.bind(('localhost', 0))    
    port = s.getsockname()[1]    
    s.close()    
    return port

def init_model(model_dir,infer_params:Dict[str,str]={}): 
    sharded = infer_params.get("model.sharded",True)
    # use os.environ["CUDA_VISIBLE_DEVICES"] to get available gpus
    gpu_ids = [int(item) for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    if len(gpu_ids) == 1:
        return _init_model(model_dir,infer_params)
    
    # use torch.multiprocessing.spawn to init model in each process
    master_port = get_available_port()
    
    for rank in range(gpu_ids):
        env = {
        "RANK": rank,
        "WORLD_SIZE": str(len(gpu_ids)),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": master_port,
        "NCCL_BLOCKING_WAIT": "1",
        "SAFETENSORS_FAST_GPU":"1"
    }
        mp.spawn(_init_model, args=(model_dir,infer_params), 
                 nprocs=len(gpu_ids), 
                 join=True,
                 env=env)



        
   