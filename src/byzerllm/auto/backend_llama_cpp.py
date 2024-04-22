import os
import time
import ray
import asyncio
from typing import List, Dict, Any,Union
import traceback
import uuid
import threading
import json
import inspect

from llama_cpp import Llama
import llama_cpp
from byzerllm.utils import ( 
    BlockVLLMStreamServer,   
    StreamOutputs,
    SingleOutput,
    SingleOutputMeta,    
)

def get_bool(params:Dict[str,str],key:str,default:bool=False)->bool:
    if key in params:
        if isinstance(params[key],bool):
            return params[key]
        else:            
            return params[key] == "true" or params[key] == "True"
    return default


def convert_params(params: Dict[str, str]) -> Dict[str, Any]:
    converted_params = {
        "model_path": params.get("model_path", ""),
        "n_gpu_layers": int(params.get("n_gpu_layers", 0)),
        "split_mode": int(params.get("split_mode", llama_cpp.LLAMA_SPLIT_MODE_LAYER)),
        "main_gpu": int(params.get("main_gpu", 0)),
        "tensor_split": eval(params.get("tensor_split", "None")),
        "vocab_only": eval(params.get("vocab_only", "False")),
        "use_mmap": eval(params.get("use_mmap", "True")),
        "use_mlock": eval(params.get("use_mlock", "False")),
        "kv_overrides": eval(params.get("kv_overrides", "None")),
        "seed": int(params.get("seed", llama_cpp.LLAMA_DEFAULT_SEED)),
        "n_ctx": int(params.get("n_ctx", 512)),
        "n_batch": int(params.get("n_batch", 512)),
        "n_threads": eval(params.get("n_threads", "None")),
        "n_threads_batch": eval(params.get("n_threads_batch", "None")),
        "rope_scaling_type": eval(params.get("rope_scaling_type", "llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED")),
        "pooling_type": int(params.get("pooling_type", llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED)),
        "rope_freq_base": float(params.get("rope_freq_base", 0.0)),
        "rope_freq_scale": float(params.get("rope_freq_scale", 0.0)),
        "yarn_ext_factor": float(params.get("yarn_ext_factor", -1.0)),
        "yarn_attn_factor": float(params.get("yarn_attn_factor", 1.0)),
        "yarn_beta_fast": float(params.get("yarn_beta_fast", 32.0)),
        "yarn_beta_slow": float(params.get("yarn_beta_slow", 1.0)),
        "yarn_orig_ctx": int(params.get("yarn_orig_ctx", 0)),
        "logits_all": eval(params.get("logits_all", "False")),
        "embedding": eval(params.get("embedding", "False")),
        "offload_kqv": eval(params.get("offload_kqv", "True")),
        "last_n_tokens_size": int(params.get("last_n_tokens_size", 64)),
        "lora_base": params.get("lora_base", None),
        "lora_scale": float(params.get("lora_scale", 1.0)),
        "lora_path": params.get("lora_path", None),
        "numa": eval(params.get("numa", "False")),
        "chat_format": params.get("chat_format", None),
        "chat_handler": eval(params.get("chat_handler", "None")),
        "draft_model": eval(params.get("draft_model", "None")),
        "tokenizer": eval(params.get("tokenizer", "None")),
        "type_k": eval(params.get("type_k", "None")),
        "type_v": eval(params.get("type_v", "None")),
        "verbose": eval(params.get("verbose", "True"))
    }
    return converted_params

class LlamaCppBackend:

    def __init__(self,model_path, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}):
        targets = convert_params(infer_params)
        self.model = Llama(model_path=model_path,**targets)        
        self.meta = {
            "model_deploy_type": "saas",
            "backend":"llama_cpp",
            "support_stream": True,            
        }

        try:
            ray.get_actor("BLOCK_VLLM_STREAM_SERVER") 
        except ValueError:  
            try:          
                ray.remote(BlockVLLMStreamServer).options(name="BLOCK_VLLM_STREAM_SERVER",lifetime="detached",max_concurrency=1000).remote()
            except Exception as e:
                pass


    def get_meta(self):
        return [self.meta]

    def process_input(self, ins: Union[str, List[Dict[str, Any]]]):
        
        if isinstance(ins, list):
            return ins
        
        content = []
        try:
            ins_json = json.loads(ins)
        except:            
            return ins
        
        content = []
        for item in ins_json:
            if "image" in item or "image_url" in item:
                image_data = item.get("image",item.get("image_url",""))
                ## "data:image/jpeg;base64," 
                if not image_data.startswith("data:"):
                    image_data = "data:image/jpeg;base64," + image_data                                                                                
                content.append({"image_url": {"url":image_data},"type": "image_url",})
            elif "text" in item:
                text_data = item["text"]
                content.append({"text": text_data,"type":"text"})
        if not content:
            return ins
        
        return content   
        
    async def async_get_meta(self):
        return self.get_meta()

    async def async_stream_chat(self, tokenizer, ins: str, his: List[Dict[str, str]] = [],
                 max_length: int = 4090, top_p: float = 0.95, temperature: float = 0.1, **kwargs):
        return await self.generate(tokenizer=tokenizer, ins=ins, his=his, max_length=max_length, top_p=top_p, temperature=temperature, **kwargs)
    
    def embed_query(self, ins: str, **kwargs):                     
        resp = self.model.create_embedding(input = [ins])
        embedding = resp.data[0].embedding
        usage = resp.usage
        return (embedding,{"metadata":{
                "input_tokens_count":usage.prompt_tokens,
                "generated_tokens_count":0}})            
   
    async def generate(self, tokenizer, ins: str, his: List[Dict[str, str]] = [],
                 max_length: int = 4090, top_p: float = 0.95, temperature: float = 0.1, **kwargs):
                           
        messages = [{"role":message["role"],"content":self.process_input(message["content"])} for message in his] + [{"role": "user", "content": self.process_input(ins)}]

        stream = kwargs.get("stream",False)
        
        server = ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
        request_id = [None]
        
        def writer():
            try:
                r = ""       
                response = self.model.create_chat_completion_openai_v1(
                                    messages=messages,                                    
                                    stream=True, 
                                    max_tokens=max_length,
                                    temperature=temperature,
                                    top_p=top_p                                                                        
                                )                                    
                request_id[0] = str(uuid.uuid4())                

                for chunk in response:                                                              
                    content = chunk.choices[0].delta.content or ""
                    r += content        
                    if hasattr(chunk,"usage"):
                        input_tokens_count = chunk.usage.prompt_tokens
                        generated_tokens_count = chunk.usage.completion_tokens
                    else:
                        input_tokens_count = 0
                        generated_tokens_count = 0
                    ray.get(server.add_item.remote(request_id[0], 
                                                    StreamOutputs(outputs=[SingleOutput(text=r,metadata=SingleOutputMeta(
                                                        input_tokens_count=input_tokens_count,
                                                        generated_tokens_count=generated_tokens_count,
                                                    ))])
                                                    ))                                                   
            except:
                traceback.print_exc()            
            ray.get(server.mark_done.remote(request_id[0]))

        if stream:
            threading.Thread(target=writer,daemon=True).start()            
                            
            time_count= 10*100
            while request_id[0] is None and time_count > 0:
                time.sleep(0.01)
                time_count -= 1
            
            if request_id[0] is None:
                raise Exception("Failed to get request id")
            
            def write_running():
                return ray.get(server.add_item.remote(request_id[0], "RUNNING"))
                        
            await asyncio.to_thread(write_running)
            return [("",{"metadata":{"request_id":request_id[0],"stream_server":"BLOCK_VLLM_STREAM_SERVER"}})]
        else:
            try:
                start_time = time.monotonic()
                response = self.model.create_chat_completion_openai_v1(
                                    messages=messages,                                    
                                    max_tokens=max_length,
                                    temperature=temperature,
                                    top_p=top_p                            
                                )

                generated_text = response.choices[0].message.content
                generated_tokens_count = response.usage.completion_tokens
                input_tokens_count = response.usage.prompt_tokens
                time_cost = time.monotonic() - start_time
                return [(generated_text,{"metadata":{
                            "request_id":response.id,
                            "input_tokens_count":input_tokens_count,
                            "generated_tokens_count":generated_tokens_count,
                            "time_cost":time_cost,
                            "first_token_time":0,
                            "speed":float(generated_tokens_count)/time_cost,        
                        }})]
            except Exception as e:
                print(f"Error: {e}")
                raise e