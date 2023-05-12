from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from typing import Any, List, Mapping, Optional,Tuple
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import requests
import json
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dataclasses import dataclass
from pathlib import Path
import ray
import os
from typing import Dict


@dataclass
class ClientParams:
    owner:str="admin"
    llm_func: str = "chat"    


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
            raise Exception(f"{self.url} status:{response.status_code} content: {response.text}")
        return response.text

    def emb(self,s:str)-> List[float]:
        json_data = json.dumps([
            {"instruction":s,"embedding":True}
        ])
        response = self.request(f'''
        select {self.client_params.llm_func}(array(feature)) as value
        ''',json_data)    
        t = json.loads(response)
        t2 = json.loads(t[0]["value"][0])
        return t2[0]["predict"]

    def chat(self,s:str,history:List[Tuple[str,str]])->str:
        newhis = [{"query":item[0],"response":item[1]} for item in history]
        json_data = json.dumps([
            {"instruction":s,"history":newhis,"output":"NAN"}
        ])
        response = self.request(f'''
        select {self.client_params.llm_func}(array(feature)) as value
        ''',json_data)    
        t = json.loads(response)
        t2 = json.loads(t[0]["value"][0])
        return t2[0]["predict"]


class LocalEmbeddings(Embeddings):
    def __init__(self,client:ByzerLLMClient):
        self.client = client
                
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:        
        embeddings = [self.client.emb(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:    
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


class VectorDB:
    def __init__(self,db_dir:str,client:ByzerLLMClient) -> None:
        self.db_dir = db_dir 
        self.db = None  
        self.client = client 
        self.embeddings = LocalEmbeddings(self.client)     
    
    def _is_visible(self,p: Path) -> bool:
        parts = p.parts
        for _p in parts:
            if _p.startswith("."):
                return False
        return True
    
    def save(self,path):        
        p = Path(path)
        docs = []
        items = list(p.rglob("**/*.json"))
        for i in items:
            if i.is_file():
               with open(i.as_posix(),"r",encoding="utf-8") as f:
                 doc = json.loads(f.readline())
                 docs.append(Document(page_content=doc["page_content"],metadata={ "source":doc["source"]}))
        
        text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=30)
        split_docs = text_splitter.split_documents(docs)                
        db = FAISS.from_documents(split_docs, self.embeddings)
        db.save_local(self.db_dir) 

    def query(self,prompt:str,s:str)->List[Document]:
        if not self.db:
           self.db = FAISS.load_local(self.db_dir,self.embeddings)        
        result = self.db.similarity_search(prompt + s)
        return result

class ByzerLLMQA:
    def __init__(self,db_dir:str,client:ByzerLLMClient) -> None:
        self.db_dir = db_dir
        self.client = client
        self.db = VectorDB(self.db_dir,self.client)

    def query(self,prompt:str,q:str):        
        docs = self.db.query("",q)
        newq = "".join([doc.page_content for doc in docs])
        if prompt == "show query":
            print(newq)
            prompt = ""
        v = self.client.chat(prompt + newq + q,[])
        return v

    def predict(self,input:Dict[str,Any]):        
        q = input["instruction"]
        prompt = input.get("prompt","")
        return self.query(prompt,q)
    

@ray.remote
class RayByzerLLMQA:
    def __init__(self,db_dir:str,client:ByzerLLMClient) -> None:
        self.db_dir = db_dir
        self.client = client        
    
    def save(self,data_refs):
        from pyjava.api.mlsql import RayContext
        from pyjava.storage import streaming_tar
        import uuid
        import shutil
        data_path = "/tmp/model/{}".format(str(uuid.uuid4()))
        
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            
        with open(os.path.join(data_path,"data.json"),"w",encoding="utf-8") as f:
            for item in RayContext.collect_from(data_refs):
                f.write(json.dumps(item,ensure_ascii=False)+"\n")

        db = VectorDB(self.db_dir,self.client)
        db.save(data_path)
        return list(streaming_tar.build_rows_from_file(self.db_dir))






