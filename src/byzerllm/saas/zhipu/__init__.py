from zhipuai import ZhipuAI
import time
from typing import List, Tuple, Dict,Any
import ray
from byzerllm.utils.types import BlockVLLMStreamServer,StreamOutputs,SingleOutput,SingleOutputMeta
from byzerllm.utils.langutil import asyncfy_with_semaphore
import threading
import asyncio


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key = infer_params["saas.api_key"]
        # chatglm_lite, chatglm_std, chatglm_pro
        self.model = infer_params.get("saas.model", "glm-4")        
        self.client = ZhipuAI(api_key=self.api_key) 

        self.meta = {
            "model_deploy_type": "saas",
            "backend":"saas",
            "support_stream": True
        }

        if "embedding" not in  self.model.lower():            
            self.meta["embedding_mode"] = False 
        else:            
            self.meta["embedding_mode"] = True       
        try:
            ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
        except ValueError:            
            ray.remote(BlockVLLMStreamServer).options(name="BLOCK_VLLM_STREAM_SERVER",lifetime="detached",max_concurrency=1000).remote()            

    # saas/proprietary
    def get_meta(self):
        return [self.meta]

    async def async_get_meta(self):
        return await asyncfy_with_semaphore(self.get_meta)() 

    def embed_query(self, ins: str, **kwargs):                     
        start_time = time.monotonic()
        response = self.client.embeddings.create(
                model=self.model,
                input=ins,
            )
        time_cost = time.monotonic() - start_time
        return response.data[0].embedding
    
    async def async_embed_query(self, ins: str, **kwargs):
        return await asyncfy_with_semaphore(self.embed_query)(ins, **kwargs)

    async def async_stream_chat(self, tokenizer, ins: str, his: List[Dict[str, Any]] = [],
                    max_length: int = 4096,
                    top_p: float = 0.7,
                    temperature: float = 0.9, **kwargs):
        
        messages = his + [{"role": "user", "content": ins}]
        
        stream = kwargs.get("stream",False)    

        other_params = {}
        
        if "stream" in kwargs:        
            other_params["stream"] = kwargs["stream"]

        for k, v in kwargs.items():
            if k in ["max_tokens", "stop"]:
                other_params[k] = v
        
        start_time = time.monotonic()
        res_data = await asyncfy_with_semaphore(lambda:self.client.chat.completions.create(
                            model=self.model,
                            temperature = temperature,
                            top_p = top_p,
                            messages=messages,**other_params))()
        
        if stream:            
            server = ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
            request_id = [None]

            def writer(): 
                r = ""
                for response in res_data:                                        
                    v = response.choices[0].delta.content
                    r += v
                    request_id[0] = f"zhipu_{response.id}"
                    ray.get(server.add_item.remote(request_id[0], 
                                                    StreamOutputs(outputs=[SingleOutput(text=r,metadata=SingleOutputMeta(
                                                        input_tokens_count= -1,
                                                        generated_tokens_count= -1,
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
        generated_text = res_data.choices[0].message.content        
        generated_tokens_count = res_data.usage.completion_tokens

        return [(generated_text,{"metadata":{
                        "request_id":res_data.id,
                        "input_tokens_count":res_data.usage.prompt_tokens,
                        "generated_tokens_count":generated_tokens_count,
                        "time_cost":time_cost,
                        "first_token_time":0,
                        "speed":float(generated_tokens_count)/time_cost,        
                    }})]


