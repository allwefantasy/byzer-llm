import json

from langchain import PromptTemplate
import ray
import os
from typing import Dict, List, Any, Tuple
import uuid
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from pyjava.storage  import streaming_tar
import time

from . import BuilderParams,QueryParams
from .qa_strategy import FullDocCombineFormatFactory, DocRetrieveStrategyFactory
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

    def query(self,prompt:str,q:str,k=4,hint="",input:Dict[str,Any]={}): 
         
        docs_with_score = [] 

        time1 = time.time()
        submit_q = [db.query.remote("",q,k) for db in self.dbs]        
        for q_func in submit_q:            
            t = ray.get(q_func)
            docs_with_score = docs_with_score + t

        query_vector_db_time = time.time() - time1        
        print(f"VectorDB query time taken:{query_vector_db_time}s. total chunks: {len(docs_with_score)}")   

        docs = sorted(docs_with_score, key=lambda doc: doc[1],reverse=False)

        strategy = input.get("strategy", "")
        docs = DocRetrieveStrategyFactory(strategy).retrieve(docs, k)

        if hint == "show_only_context":
            return json.dumps([{"score":float(doc[1]),"content":doc[0].page_content} for doc in docs[0:k]],ensure_ascii=False,indent=4)
         
        newq , temp_metas = FullDocCombineFormatFactory(input).combine(docs, k)
        
        show_full_query  = hint == "show_full_query"         

        if not prompt:
            prompt = "{context} \n {query}"

        prompt_template = PromptTemplate.from_template(prompt)
 
        final_query = prompt_template.format(context=newq, query=q)

        time2 = time.time()
        
        top_p=float(input.get("top_p",0.7))
        temperature=float(input.get("temperature",0.9))
        max_length = int(input.get("max_length", 1024))

        v = self.client.chat(final_query,[],extra_query={"top_p":top_p,"temperature":temperature, "max_length":max_length})
        chat_time = time.time() - time2

        if show_full_query:
          return json.dumps({
             "query": final_query,
             "predict": v,
             "vectordb_time": f"{query_vector_db_time}s",
             "chat_time": f"{chat_time}s",
             "metas":temp_metas
          },ensure_ascii=False,indent=4)
        else:
          return v

    def predict(self,input:Dict[str,Any]):        
        q = input["instruction"]
        prompt = input.get("prompt","")
        hint = input.get("hint","")
        k = int(input.get("k","4"))
        return self.query(prompt,q,k,hint,input)

@ray.remote
class RayByzerLLMQAWorker: 
    def __init__(self,data_ref,client:ByzerLLMClient) -> None:
        self.data_ref = data_ref        
        self.client = client         
    
    def build(self,params:BuilderParams,extra_params={}):
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
        db = VectorDB(db_dir,self.client,extra_params=extra_params)
        db.build_from(data_path,params,extra_params=extra_params)
        
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
    
    def save(self,data_refs,params=BuilderParams(),builder_params={}):        
        from pyjava.storage import streaming_tar     

        self.db_dir = os.path.join(params.local_path_prefix,"qa",str(uuid.uuid4()))

        data = []
        workers = []

        ## build vector db file in parallel
        print(f"Start {len(data_refs)} workers to build vector db")
        for data_ref in data_refs:            
            worker = RayByzerLLMQAWorker.remote(data_ref,self.client)
            workers.append(worker)
            build_func = worker.build.remote(params,builder_params)
            data.append(build_func)

        ## gather all db file and merge into one
        print(f"gather all db file and merge into one {self.db_dir}")
        for index, build_func in enumerate(data):
            sub_data = ray.get(build_func)
            temp_dir = os.path.join(self.db_dir,f"vecdb_{index}")
            streaming_tar.save_rows_as_file((ray.get(ref) for ref in sub_data),temp_dir)                     
        
        return streaming_tar.build_rows_from_file(self.db_dir)
        






