
from typing import List, Tuple, Dict,Any
import httpx
from openai import OpenAI   

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
                    

        self.proxies = infer_params.get("saas.proxies", None)
        self.local_address = infer_params.get("saas.local_address", "0.0.0.0")
                
        
        if self.proxies is None or self.proxies == "":
            self.client = OpenAI(**other_params,api_key=self.api_key)  
        else:
            self.client = OpenAI(**other_params,api_key=self.api_key,http_client=httpx.Client(
                proxies=self.proxies,
                transport=httpx.HTTPTransport(local_address=self.local_address)))                        
    
        

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

        model = self.model
        max_retries = self.max_retries

        if "model" in kwargs:
            model = kwargs["model"]
        if "max_retries" in kwargs:
            max_retries = kwargs["max_retries"]

        messages = his + [{"role": "user", "content": ins}]
        
        try:
            response = self.client.chat.completions.create(
                                messages=messages,
                                model=model,
                                max_tokens=max_length,
                                temperature=temperature,
                                top_p=top_p                            
                            )

            res = response.choices[0].message.content
            return [(res, "")]
        except Exception as e:
            print(f"Error: {e}")
            raise e
