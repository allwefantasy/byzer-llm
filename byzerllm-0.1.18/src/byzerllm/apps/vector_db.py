from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from pathlib import Path
from typing import List



from . import BuilderParams
from .builder import OnceWay,MergeWay
from .client import ByzerLLMClient,LocalEmbeddings

class VectorDB:
    def __init__(self,db_dir:str,client:ByzerLLMClient,extra_params={}) -> None:
        self.db_dir = db_dir 
        self.db = None  
        self.client = client 
        self.embeddings = LocalEmbeddings(self.client,extra_params.get("promptPrefix",None))     
    
    def _is_visible(self,p: Path) -> bool:
        parts = p.parts
        for _p in parts:
            if _p.startswith("."):
                return False
        return True

    
    def save(self,path,params:BuilderParams,extra_params={}):                        
        if params.batch_size == 0:
            b = OnceWay(self.db_dir,self.embeddings)
            b.build(path,params,extra_params)
        else:
            b = MergeWay(self.db_dir,self.embeddings)
            b.build(path,params,extra_params)    
            

    def build_from(self,path,params:BuilderParams,extra_params={}):
        return self.save(path,params,extra_params) 


    def merge_from(self,target_path:str):
        if self.db is None:
            self.db = FAISS.from_documents([], self.embeddings)            

        self.db.merge_from(FAISS.load_local(target_path,self.embeddings))                

    def query(self,prompt:str,s:str,k=4)->List[Document]:
        if not self.db:
           self.db = FAISS.load_local(self.db_dir,self.embeddings)        
        result = self.db.similarity_search_with_score(prompt + s,k=k)
        return result