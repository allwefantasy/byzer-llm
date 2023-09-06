from langchain.embeddings.base import Embeddings
from typing import Any, List, Mapping, Optional,Tuple
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json

from . import ClientParams


class ByzerLLMClient:
    
    def __init__(self,url:str='http://127.0.0.1:9003/model/predict',params:ClientParams=ClientParams()) -> None:
        self.url = url
        self.client_params = params        

    def request(self, sql:str,json_data:str)->str:         
        data = {
            'sessionPerUser': 'true',
            'sessionPerRequest': 'true',
            'owner': self.client_params.owner,
            'dataType': 'string',
            'sql': sql,
            'data': json_data
        }
        response = requests.post(self.url, data=data)
        if response.status_code != 200:
            raise Exception(f"{self.url} status:{response.status_code} content: {response.text} request: json/{json.dumps(data,ensure_ascii=False)}")
        return response.text

    def emb(self,s:str)-> List[float]:
        json_data = json.dumps([
            {"instruction":s,"embedding":True}
        ],ensure_ascii=False)
        response = self.request(f'''
        select {self.client_params.llm_embedding_func}(array(feature)) as value
        ''',json_data)    
        t = json.loads(response)
        t2 = json.loads(t[0]["value"][0])
        return t2[0]["predict"]

    def chat(self,s:str,history:List[Tuple[str,str]],extra_query={})->str:
        newhis = [{"query":item[0],"response":item[1]} for item in history]
        json_data = json.dumps([
            {"instruction":s,"history":newhis,**extra_query}
        ],ensure_ascii=False)
        response = self.request(f'''
        select {self.client_params.llm_chat_func}(array(feature)) as value
        ''',json_data)    
        t = json.loads(response)
        t2 = json.loads(t[0]["value"][0])
        return t2[0]["predict"]


class LocalEmbeddings(Embeddings):
    def __init__(self,client:ByzerLLMClient,prompt_prefix=None):
        self.client = client
        self.prompt_prefix = prompt_prefix
                
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []        
        for text in texts:
            if self.prompt_prefix:
                text = self.prompt_prefix + text
            embedding = self.client.emb(text)
            embeddings.append(embedding)        
        return embeddings

    def embed_query(self, text: str) -> List[float]: 
        if self.prompt_prefix:
            text = self.prompt_prefix + text   
        embedding = self.client.emb(text)
        return embedding


class Chatglm6bLLM(LLM):
    
    def __init__(self,client:ByzerLLMClient):
        self.client = client
        
    @property
    def _llm_type(self) -> str:
        return "chatglm6b"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.client.chat(prompt,[])
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}