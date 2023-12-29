import requests
import json
import time
import hashlib
import traceback
from retrying import retry
from typing import List, Tuple, Dict,Any

BaiChuanErrorCodes = {
    "0": "success",
    "1": "system error",
    "10000": "Invalid parameters, please check",
    "10100": "Missing apikey",
    "10101": "Invalid apikey",
    "10102": "apikey has expired",
    "10103": "Invalid Timestamp parameter in request header",
    "10104": "Expire Timestamp parameter in request header",
    "10105": "Invalid Signature parameter in request header",
    "10106": "Invalid encryption algorithm in request header, not supported by server",
    "10200": "Account not found",
    "10201": "Account is locked, please contact the support staff",
    "10202": "Account is temporarily locked, please try again later",
    "10203": "Request too frequent, please try again later",
    "10300": "Insufficient account balance, please recharge",
    "10301": "Account is not verified, please complete the verification first",
    "10400": "Topic violates security policy for prompts",
    "10401": "Topic violates security policy for answer",
    "10500": "Internal error",
}


class CustomSaasAPI:
    def __init__(self, infer_params: Dict[str, str]) -> None:
        self.api_key = infer_params["saas.api_key"]
        self.secret_key = infer_params.get("saas.secret_key","")
        self.api_url = infer_params.get("saas.baichuan_api_url", "https://api.baichuan-ai.com/v1/chat/completions")
        self.model = infer_params.get("saas.model", "Baichuan2-Turbo")

     # saas/proprietary
    def get_meta(self):
        return [{
            "model_deploy_type": "saas",
            "backend":"saas"
        }]
    
    def stream_chat(self, tokenizer, ins: str, his: List[Dict[str, Any]] = [],
                    max_length: int = 4096,
                    top_p: float = 0.9,
                    temperature: float = 0.1, **kwargs):
        
        messages = his + [{"role": "user", "content": ins}]

        other_params = {}
        if "with_search_enhance" in kwargs:
            other_params["with_search_enhance"] = kwargs["with_search_enhance"]
        
        if "top_k" in kwargs:
            other_params["top_k"] = kwargs["top_k"]

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            **other_params
        }        
        
        start_time = time.monotonic()
        res_data = self.request_with_retry(data)   
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
        

    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=3)
    def request_with_retry(self, data):
        json_data = json.dumps(data)
        # time_stamp = int(time.time())
        # signature = self.calculate_md5(self.secret_key + json_data + str(time_stamp))
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            # "X-BC-Timestamp": str(time_stamp),
            # "X-BC-Signature": signature,
            # "X-BC-Sign-Algo": "MD5",
        }
        response = requests.post(self.api_url, data=json_data, headers=headers)
        if response.status_code == 200:
            # id = response.headers.get("X-BC-Request-Id")
            
            res_data = json.loads(response.text)
            
            if "code" in res_data and  res_data["code"] != 0:
                if str(res_data["code"]) in BaiChuanErrorCodes.keys():
                    msg = BaiChuanErrorCodes.get(str(res_data["code"]))
                    print("request baichuan api failed, err msg:" + msg)
                    # check if api call over limit
                    if str(res_data["code"]) == "10203":
                        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print("{0} request baichuan api too frequent, retrying... ".format(now))
                        raise Exception("request baichuan api too frequent." + str(res_data["code"]))
                    else:
                        return msg
                # unexpected error code, retry anyway
                raise Exception("request baichuan api failed, api response code:" + str(res_data["code"]))                        
        
            return res_data


        else:
            print("request baichuan api failed, http response code:" + str(response.status_code))
            print("response text:" + response.text)
            raise Exception("request baichuan api failed")

    def calculate_md5(self, input_string):
        md5 = hashlib.md5()
        md5.update(input_string.encode('utf-8'))
        encrypted = md5.hexdigest()
        return encrypted

