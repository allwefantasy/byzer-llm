from typing import List, Dict
import time
import qianfan
import threading
import asyncio
import ray

from byzerllm.utils import random_uuid
from byzerllm.log import init_logger
from byzerllm.utils.types import BlockVLLMStreamServer, StreamOutputs, SingleOutput, SingleOutputMeta
from byzerllm.utils.langutil import asyncfy_with_semaphore

logger = init_logger(__name__)


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key: str = infer_params.get("saas.api_key", "")
        self.secret_key: str = infer_params.get("saas.secret_key", "")
        self.access_token: str = infer_params.get("saas.access_token", "")

        if not self.access_token and (not self.api_key or not self.secret_key):
            raise ValueError("Please specify either access_token or ak/sk")

        # os.environ["QIANFAN_ACCESS_KEY"] = self.api_key
        # os.environ["QIANFAN_SECRET_KEY"] = self.secret_key
        # qianfan.AK(self.api_key)
        # qianfan.SK(self.secret_key)
        self.model: str = infer_params.get("saas.model", "ERNIE-Bot-turbo")
        self.client = qianfan.ChatCompletion(ak=self.api_key, sk=self.secret_key, access_token=self.access_token)
        try:
            ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
        except ValueError:            
            ray.remote(BlockVLLMStreamServer).options(name="BLOCK_VLLM_STREAM_SERVER",lifetime="detached",max_concurrency=1000).remote() 

     # saas/proprietary
    def get_meta(self):
        return [{
            "model_deploy_type": "saas",
            "backend":"saas",
            "support_stream": True
        }] 

    async def async_get_meta(self):
        return await asyncfy_with_semaphore(self.get_meta)()   

    async def async_stream_chat(
            self,
            tokenizer,
            ins: str,
            his: List[dict] = [],
            max_length: int = 4096,
            top_p: float = 0.7,
            temperature: float = 0.9,
            **kwargs
    ):
        request_id = kwargs.get("request_id", random_uuid())
        other_params = {}
        
        if "request_timeout" in kwargs:
            other_params["request_timeout"] = int(kwargs["request_timeout"])
        
        if "retry_count" in kwargs:
            other_params["retry_count"] = int(kwargs["retry_count"])
        
        if "backoff_factor" in kwargs:
            other_params["backoff_factor"] = float(kwargs["backoff_factor"])  

        if "penalty_score" in kwargs:
            other_params["penalty_score"] = float(kwargs["penalty_score"])  

        stream = kwargs.get("stream",False)                                  

        messages = qianfan.Messages()
        for item in his:
            role, content = item['role'], item['content']
            # messages must have an odd number of members
            # look for details: https://cloud.baidu.com/doc/WENXINWORKSHOP/s/clntwmv7t
            if role == 'system':
                messages.append(content, qianfan.Role.User)
                messages.append("OK", qianfan.Role.Assistant)
                continue
            messages.append(content, role)

        if ins:
            messages.append(ins, qianfan.Role.User)
        
        start_time = time.monotonic()

        logger.info(f"Receiving request {request_id} model: {self.model}")

        res_data = await asyncfy_with_semaphore(lambda:self.client.do(
            model=self.model,            
            messages=messages,
            top_p=top_p,
            temperature=temperature,
            stream=stream,
            **other_params
        ))()
        
        if stream:
            server = ray.get_actor("BLOCK_VLLM_STREAM_SERVER")
            request_id = [None]

            def writer(): 
                for response in res_data:                                        
                    if response["code"] == 200:
                        v = response["result"]
                        request_id[0] = f'qianfan_{response["id"]}'
                        ray.get(server.add_item.remote(request_id[0], 
                                                       StreamOutputs(outputs=[SingleOutput(text=v,metadata=SingleOutputMeta(
                                                           input_tokens_count=response["usage"]["prompt_tokens"],
                                                           generated_tokens_count=response["usage"]["completion_tokens"],
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

        generated_text = res_data["result"]
        generated_tokens_count = res_data["usage"]["completion_tokens"]
        input_tokens_count = res_data["usage"]["prompt_tokens"]

        logger.info(
            f"Completed request {request_id} "
            f"model: {self.model} "
            f"cost: {time_cost} "
            f"result: {res_data}"
        )

        return [(generated_text,{"metadata":{
                "request_id":res_data["id"],
                "input_tokens_count":input_tokens_count,
                "generated_tokens_count":generated_tokens_count,
                "time_cost":time_cost,
                "first_token_time":0,
                "speed":float(generated_tokens_count)/time_cost,        
            }})] 