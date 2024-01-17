import json
from zhipuai import ZhipuAI
import time
import traceback
from typing import List, Tuple, Dict,Any
from retrying import retry


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key = infer_params["saas.api_key"]
        # chatglm_lite, chatglm_std, chatglm_pro
        self.model = infer_params.get("saas.model", "glm-4")        
        self.client = ZhipuAI(api_key=self.api_key)         

    # saas/proprietary
    def get_meta(self):
        return [{
            "model_deploy_type": "saas",
            "backend":"saas"
        }]    

    def stream_chat(self, tokenizer, ins: str, his: List[Dict[str, Any]] = [],
                    max_length: int = 4096,
                    top_p: float = 0.7,
                    temperature: float = 0.9, **kwargs):
        
        messages = his + [{"role": "user", "content": ins}]

        other_params = {}

        for k, v in kwargs.items():
            if k in ["max_tokens", "stop"]:
                other_params[k] = v
                
        
        start_time = time.monotonic()
        res_data = self.client.chat.completions.create(
                            model=self.model,
                            temperature = temperature,
                            top_p = top_p,
                            messages=messages,**other_params)
      
        time_cost = time.monotonic() - start_time
        generated_text = res_data["choices"][0]["message"]["content"] 

        generated_tokens_count = res_data["usage"]["completion_tokens"]   

        return [(generated_text,{"metadata":{
                        "request_id":res_data["id"],
                        "input_tokens_count":res_data["usage"]["prompt_tokens"],
                        "generated_tokens_count":generated_tokens_count,
                        "time_cost":time_cost,
                        "first_token_time":0,
                        "speed":float(generated_tokens_count)/time_cost,        
                    }})]


