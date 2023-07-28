import openai
from typing import Union, List, Tuple, Optional, Dict


class CustomSaasAPI:
    def __init__(self,infer_params:Dict[str,str]) -> None:
         self.api_type = infer_params["saas.api_type"]
         self.api_key = infer_params["saas.api_key"]
         self.api_base = infer_params["saas.api_base"]
         self.api_version = infer_params["saas.api_version"]
         self.deployment_id = infer_params["saas.deployment_id"]
         openai.api_type = infer_params["saas.api_type"]
         openai.api_key = infer_params["saas.api_key"]
         openai.api_base = infer_params["saas.api_base"]
         openai.api_version = infer_params["saas.api_version"] 
            
    
    def stream_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096,
        top_p:float=0.7,
        temperature:float=0.9,**kwargs): 

        deployment_id=self.deployment_id
        
        if "model" in kwargs:
            deployment_id = kwargs["model"]

        his_message = []

        for item in his:
            his_message.append({"role": "user", "content": item[0]}) 
            his_message.append({"role": "assistant", "content": item[1]})   


        messages= his + [{"role": "user", "content": ins}]

        chat_completion = openai.ChatCompletion.create(messages=messages,
                                                       deployment_id=deployment_id,
                                                       temperature=temperature,
                                                       top_p=top_p,max_tokens=max_length)
        res_text = chat_completion.choices[0]["message"]["content"].replace('\n', '').replace(' .', '.').strip()
        return [(res_text,"")]





