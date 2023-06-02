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
import uuid
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from pyjava.storage  import streaming_tar
import time

@dataclass
class ClientParams:
    owner:str="admin"
    llm_func: str = "chat"  

@dataclass
class BuilderParams:
    chunk_size:int=600
    chunk_overlap: int = 30
    local_path_prefix: str = "/tmp/byzer-llm-qa-model"

@dataclass
class QueryParams:    
    local_path_prefix: str = "/tmp/byzer-llm-qa-model"    


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

    
    def save(self,path,params:BuilderParams):        
        p = Path(path)
        docs = []
        items = list(p.rglob("**/*.json"))
        for i in items:
            if i.is_file():
               with open(i.as_posix(),"r",encoding="utf-8") as f:
                 for line in f:
                    doc = json.loads(line)
                    docs.append(Document(page_content=doc["page_content"],metadata={ "source":doc["source"]}))
        
        text_splitter = CharacterTextSplitter(chunk_size=params.chunk_size, chunk_overlap=params.chunk_overlap)
        split_docs = text_splitter.split_documents(docs) 
        print(f"Build vector db in {self.db_dir}. total docs: {len(docs)} total split docs: {len(split_docs)}")               
        db = FAISS.from_documents(split_docs, self.embeddings)
        db.save_local(self.db_dir) 

    def build_from(self,path,params:BuilderParams):
        return self.save(path,params) 


    def merge_from(self,target_path:str):
        if self.db is None:
            self.db = FAISS.from_documents([], self.embeddings)            

        self.db.merge_from(FAISS.load_local(target_path,self.embeddings))                

    def query(self,prompt:str,s:str,k=4)->List[Document]:
        if not self.db:
           self.db = FAISS.load_local(self.db_dir,self.embeddings)        
        result = self.db.similarity_search_with_score(prompt + s,k=k)
        return result


@ray.remote
class ByzerLLMQAQueryWorker:
    def __init__(self,refs:List[ClientObjectRef],client:ByzerLLMClient,query_param:QueryParams) -> None:        
        self.client = client        
        db_path = os.path.join(query_param.local_path_prefix,str(uuid.uuid4()))
        streaming_tar.save_rows_as_file((ray.get(ref) for ref in refs),db_path)
        self.db = VectorDB(db_path,self.client)
                
    def query(self,prompt:str,q:str,k=4):                                
        return self.db.query(prompt,q,k)        



class ByzerLLMQA:
    def __init__(self,db_dir:str,client:ByzerLLMClient,query_params:QueryParams) -> None:
        self.db_dir = db_dir
        self.client = client
        self.query_params = query_params
        self.db_dirs = [os.path.join(self.db_dir,item) for item in os.listdir(self.db_dir)]
        self.dbs = []
        for dd in self.db_dirs:
            refs = [ray.put(item) for item in streaming_tar.build_rows_from_file(dd)]
            self.dbs.append(ByzerLLMQAQueryWorker.remote(refs,self.client,self.query_params))                    

    def query(self,prompt:str,q:str,k=4): 
         
        docs_with_score = [] 

        time1 = time.time()
        submit_q = [db.query.remote("",q) for db in self.dbs]        
        for q_func in submit_q:            
            t = ray.get(q_func)
            docs_with_score = docs_with_score + t
                
        print(f"VectorDB query time taken:{time.time()-time1}s. total chunks: {len(docs_with_score)}")   

        docs = sorted(docs_with_score, key=lambda doc: doc[1],reverse=True)
        newq = "".join([doc[0].page_content for doc in docs[0:k]])        

        show_query  = prompt == "show query"
        if show_query:
            print(":all docs ====== \n")
            for doc in docs_with_score:
                print(f"score:{doc[1]} => ${doc[0].page_content}") 

            print(":merge docs ====== \n")    

            for doc in docs:
                print(f"score:{doc[1]} => ${doc[0].page_content}") 

            prompt = ""
        v = self.client.chat(prompt + newq + q,[])

        if show_query:
          return f'[prompt:]{prompt} \n [newq:]{newq} \n [q:]{q} \n  [v:]{v}'
        else:
          return v

    def predict(self,input:Dict[str,Any]):        
        q = input["instruction"]
        prompt = input.get("prompt","")
        k = int(input.get("k","4"))
        return self.query(prompt,q,k)

@ray.remote
class RayByzerLLMQAWorker: 
    def __init__(self,data_ref,client:ByzerLLMClient) -> None:
        self.data_ref = data_ref        
        self.client = client         
    
    def build(self,params:BuilderParams):
        from pyjava.api.mlsql import RayContext
        from pyjava.storage import streaming_tar
        import uuid
        import shutil        

        data_path = os.path.join(params.local_path_prefix,str(uuid.uuid4()))
        
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            
        with open(os.path.join(data_path,"data.json"),"w",encoding="utf-8") as f:
            for item in RayContext.collect_from([self.data_ref]):
                f.write(json.dumps(item,ensure_ascii=False)+"\n")
        
        
        db_dir = os.path.join(params.local_path_prefix,str(uuid.uuid4()))
        db = VectorDB(db_dir,self.client)
        db.build_from(data_path,params)
        
        refs = []
        for item in  streaming_tar.build_rows_from_file(db_dir):
            item_ref = ray.put(item)
            refs.append(item_ref)

        shutil.rmtree(data_path,ignore_errors=True)
        shutil.rmtree(db_dir)
        return refs

# QA Vector Store Builder
class RayByzerLLMQA:
    def __init__(self,db_dir:str,client:ByzerLLMClient) -> None:
        self.db_dir = db_dir
        self.client = client        
    
    def save(self,data_refs,params=BuilderParams()):        
        from pyjava.storage import streaming_tar                                      
        data = []
        workers = []

        ## build vector db file in parallel
        print(f"Start {len(data_refs)} workers to build vector db")
        for data_ref in data_refs:            
            worker = RayByzerLLMQAWorker.remote(data_ref,self.client)
            workers.append(worker)
            build_func = worker.build.remote(params)
            data.append(build_func)

        ## gather all db file and merge into one
        print(f"gather all db file and merge into one {self.db_dir}")
        for index, build_func in enumerate(data):
            sub_data = ray.get(build_func)
            temp_dir = os.path.join(self.db_dir,f"vecdb_{index}")
            streaming_tar.save_rows_as_file((ray.get(ref) for ref in sub_data),temp_dir)                     
                                                                               
        return streaming_tar.build_rows_from_file(self.db_dir)






