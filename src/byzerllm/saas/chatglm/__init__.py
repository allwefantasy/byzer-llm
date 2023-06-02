from wudao.api_request import executeEngineV2, getToken, queryTaskResult
import datetime
import time
import uuid

from typing import Union, List, Tuple, Optional, Dict

class ChatGLMAPI:
    def __init__(self,api_key:str, public_key:str) -> None:
        self.api_key = api_key
        self.public_key = public_key
        self.ability_type = "completions"
        self.engine_type = "completions_130B"
        self.temp_token = None

    def get_token_or_refresh(self):
        token_result = getToken(self.api_key, self.public_key)
        if token_result and token_result["code"] == 200:
            token = token_result["data"]
            self.temp_token = token
        else:
            raise Exception("Fail to get token from ChatGLMAPI. Check api_key/public_key")    
        return self.temp_token    
    
    def stream_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1):  
        data = {
                    "topP": top_p,
                    "temperature": temperature,
                    "lengthPenalty": 1,
                    "numBeams": 1,
                    "minGenLength": 50,
                    "requestTaskNo": str(uuid.uuid4()),
                    "samplingStrategy": "BeamSearchStrategy",
                    "maxTokens": max_length,
                    "prompt": ins
                }    
        token = self.temp_token if self.temp_token  else self.get_token_or_refresh()
        resp = executeEngineV2(self.ability_type, self.engine_type, token, data)
        
        if resp["code"] != 200:
           token = self.get_token_or_refresh()
           resp = executeEngineV2(self.ability_type, self.engine_type, token, data)

        while resp["code"] == 200 and resp['data']['taskStatus'] == 'PROCESSING':
            print(resp)
            taskOrderNo = resp['data']['taskOrderNo']
            time.sleep(1)
            resp = queryTaskResult(token, taskOrderNo)
        print(resp)
        v = [resp['data']['outputText'][0]]
        return [(res,"") for res in v]




