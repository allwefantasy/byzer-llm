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

def convert_string_to_type(value: str, target_type):
    # Handling Boolean separately because bool("False") is True
    if target_type == bool:
        return value.lower() in ['true', '1', 't', 'y', 'yes']
    return target_type(value)

def get_init_params_and_convert(cls, **string_values):
    signature = inspect.signature(cls.__init__)
    parameters = signature.parameters
    converted_values = {}
    for name, param in parameters.items():
        if name == 'self':
            continue
        param_type = param.annotation

        if name not in string_values:
            continue        
        param_value = string_values[name]
        converted_values[name] = convert_string_to_type(param_value, param_type)
    return converted_values

class LlamaCppBackend:

    def __init__(self,model_path, infer_params: Dict[str, str] = {}, sys_conf: Dict[str, str] = {}):
        targets = get_init_params_and_convert(Llama, **infer_params)
        self.model = Llama(model_path=model_path,**targets)        
        self.meta = {
            "model_deploy_type": "saas",
            "backend":"ray/llama_cpp",
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