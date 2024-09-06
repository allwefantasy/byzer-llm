from http import HTTPStatus
from typing import List, Dict
import uuid
import google.generativeai as genai
from google.generativeai.types import content_types
import time
import ray
from byzerllm.utils import BlockVLLMStreamServer,StreamOutputs,SingleOutput,SingleOutputMeta
import threading
import asyncio
from byzerllm.utils.langutil import asyncfy_with_semaphore


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key: str = infer_params["saas.api_key"]  
        self.model = infer_params.get("saas.model", "gemini-pro")
        self.meta = {
            "model_deploy_type": "saas",
            "backend":"saas",
            "support_stream": True
        } 
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
                
        try:
            ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
        except ValueError:            
            ray.remote(BlockVLLMStreamServer).options(name="BLOCK_VLLM_STREAM_SERVER",lifetime="detached",max_concurrency=1000).remote()     

     # saas/proprietary
    def get_meta(self):
        return [self.meta] 
    
    async def async_get_meta(self):
        return await asyncfy_with_semaphore(self.get_meta)()

    async def async_stream_chat(
            self,
            tokenizer,
            ins: str,
            his: List[dict] = [],
            max_length: int = 1024,
            top_p: float = 0.9,
            temperature: float = 0.1,
            **kwargs
    ):
                
        start_time = time.monotonic()

        other_params = {}                
        if "stream" in kwargs:        
            other_params["stream"] = kwargs["stream"]

        stream = kwargs.get("stream",False)    
        
        new_messages = []

        for message in his:
            role = message["role"]
            if role == "assistant":
                role = "model"
            new_messages.append({"role":role,"content":content_types.to_content(message["content"])})
        
        new_messages.append({"role":"user","content":content_types.to_content(ins)})   
        
        res_data = await asyncfy_with_semaphore(lambda:self.client.generate_content(contents=new_messages,stream=stream))()
        
        if stream:            
            server = ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
            request_id = [None]
           
            def writer(): 
                r = ""
                for response in res_data:                                        
                    v = response.text
                    r += v
                    request_id[0] = str(uuid.uuid4())                        
                    ray.get(server.add_item.remote(request_id[0], 
                                                    StreamOutputs(outputs=[SingleOutput(text=r,metadata=SingleOutputMeta(
                                                        input_tokens_count=0,
                                                        generated_tokens_count=0,
                                                    ))])
                                                    ))
                    
                ray.get(server.mark_done.remote(request_id[0]))

            threading.Thread(target=writer,daemon=True).start()            
                               
            time_count= 10*100
            while request_id[0] is None and time_count > 0:
                await asyncio.sleep(0.01)
                time_count -= 1
            
            if request_id[0] is None:
                raise Exception("Failed to get request id")
            
            def write_running():
                return ray.get(server.add_item.remote(request_id[0], "RUNNING"))
                        
            await asyncio.to_thread(write_running)
            return [("",{"metadata":{"request_id":request_id[0],"stream_server":"BLOCK_VLLM_STREAM_SERVER"}})]  
              
        time_cost = time.monotonic() - start_time
        
        
        generated_text = res_data.text
        generated_tokens_count = 0
        input_tokens_count = 0

        return [(generated_text,{"metadata":{
                "request_id":res_data["request_id"],
                "input_tokens_count":input_tokens_count,
                "generated_tokens_count":generated_tokens_count,
                "time_cost":time_cost,
                "first_token_time":0,
                "speed":float(generated_tokens_count)/time_cost,        
            }})] 

    
        

