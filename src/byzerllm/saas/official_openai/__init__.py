import time
from typing import List, Tuple, Dict,Any,Union
import httpx
from openai import OpenAI 
import base64
import os  
import json
import ray
from byzerllm.utils import BlockVLLMStreamServer,StreamOutputs,SingleOutput,SingleOutputMeta
import threading
import asyncio
import traceback
import uuid

class CustomSaasAPI:    

    def __init__(self, infer_params: Dict[str, str]) -> None:
             
        self.api_key = infer_params["saas.api_key"]        
        self.model = infer_params.get("saas.model","gpt-3.5-turbo-1106")
        
        other_params = {}

        if "saas.api_base" in infer_params:
            other_params["api_base"] = infer_params["saas.api_base"]
        
        if "saas.api_version" in infer_params:
            other_params["api_version"] = infer_params["saas.api_version"]
        
        if "saas.api_type" in infer_params:
            other_params["api_type"] = infer_params["saas.api_type"]

        if "saas.base_url" in infer_params:
            other_params["base_url"] = infer_params["saas.base_url"]    

        if "saas.timeout" in infer_params:
            other_params["timeout"] = float(infer_params["saas.timeout"]    )
        
        self.max_retries = int(infer_params.get("saas.max_retries",10))

        self.meta = {
            "model_deploy_type": "saas",
            "backend":"saas",
            "support_stream": True,

        }
                    

        self.proxies = infer_params.get("saas.proxies", None)
        self.local_address = infer_params.get("saas.local_address", "0.0.0.0")
                
        
        if not self.proxies:
            self.client = OpenAI(**other_params,api_key=self.api_key)  
        else:
            self.client = OpenAI(**other_params,api_key=self.api_key,http_client=httpx.Client(
                proxies=self.proxies,
                transport=httpx.HTTPTransport(local_address=self.local_address)))         
    
        try:
            ray.get_actor("BLOCK_VLLM_STREAM_SERVER") 
        except ValueError:            
            ray.remote(BlockVLLMStreamServer).options(name="BLOCK_VLLM_STREAM_SERVER",lifetime="detached",max_concurrency=1000).remote()

    # saas/proprietary
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

    async def async_stream_chat(self, tokenizer, ins: str, his: List[Dict[str, Any]] = [],
                    max_length: int = 4096,
                    top_p: float = 0.7,
                    temperature: float = 0.9, **kwargs):

        model = self.model        

        if "model" in kwargs:
            model = kwargs["model"]                  

        messages = [{"role":message["role"],"content":self.process_input(message["content"])} for message in his] + [{"role": "user", "content": self.process_input(ins)}]

        stream = kwargs.get("stream",False)
        
        server = ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
        request_id = [None]
        
        def writer():
            try:
                r = ""       
                response = self.client.chat.completions.create(
                                    messages=messages,
                                    model=model,
                                    stream=True, 
                                    max_tokens=max_length,
                                    temperature=temperature,
                                    top_p=top_p                                                                        
                                )    
                # input_tokens_count = 0     
                # generated_tokens_count = 0
                
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
                response = self.client.chat.completions.create(
                                    messages=messages,
                                    model=model,
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