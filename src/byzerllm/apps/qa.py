import json
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dataclasses import dataclass
from pathlib import Path
import ray
import os
from typing import Dict,List,Any
import uuid
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from pyjava.storage  import streaming_tar
import time
import shutil

from . import BuilderParams,QueryParams
from .vector_db import VectorDB
from .client import ByzerLLMClient


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
        submit_q = [db.query.remote("",q,k) for db in self.dbs]        
        for q_func in submit_q:            
            t = ray.get(q_func)
            docs_with_score = docs_with_score + t
                
        print(f"VectorDB query time taken:{time.time()-time1}s. total chunks: {len(docs_with_score)}")   

        docs = sorted(docs_with_score, key=lambda doc: doc[1],reverse=True)                       

        if prompt == "show_only_context":
            return json.dumps(docs,ensure_ascii=False,indent=4)

        newq = "\n".join([doc[0].page_content for doc in docs[0:k]]) 
        show_full_query  = prompt == "show_full_query" 
                    
        v = self.client.chat(prompt + newq + q,[])

        if show_full_query:
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
    def __init__(self,client:ByzerLLMClient) -> None:
        self.db_dir = ""
        self.client = client        
    
    def save(self,data_refs,params=BuilderParams()):        
        from pyjava.storage import streaming_tar     

        self.db_dir = os.path.join(params.local_path_prefix,"qa",str(uuid.uuid4()))

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
        






