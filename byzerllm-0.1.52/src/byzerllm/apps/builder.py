
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any, List, Dict
import os
import json
import shutil
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from pathlib import Path
from . import BuilderParams


class OnceWay:
    def __init__(self,db_dir,embeddings) -> None:
        self.db_dir = db_dir
        self.embeddings = embeddings

    def build(self,path:str,params:BuilderParams,extra_params={}):
        p = Path(path)
        docs = []
        items = list(p.rglob("**/*.json"))                             

        for i in items:
            if i.is_file():
               with open(i.as_posix(),"r",encoding="utf-8") as f:
                 for line in f:
                    doc = json.loads(line)
                    docs.append(Document(page_content=doc["page_content"],
                                         metadata={"source": doc["source"], "page_content": doc["page_content"]}))
                    
        
        if  len(docs) > 0:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=params.chunk_size, chunk_overlap=params.chunk_overlap)
            split_docs = text_splitter.split_documents(docs)             
            print(f"Build vector db in {self.db_dir}. total docs: {len(docs)} total split docs: {len(split_docs)}")
            db = FAISS.from_documents(split_docs, self.embeddings)                                    
            db.save_local(self.db_dir)
            docs.clear()               


class MergeWay:
    def __init__(self,db_dir,embeddings) -> None:
        self.db_dir = db_dir
        self.embeddings = embeddings

    def build(self,path:str,params:BuilderParams,extra_params={}):
        p = Path(path)
        docs = []
        items = list(p.rglob("**/*.json"))

        max_doc_size = params.batch_size

        paths = []
        counter = 0        

        for i in items:
            if i.is_file():
               with open(i.as_posix(),"r",encoding="utf-8") as f:
                 for line in f:
                    doc = json.loads(line)                    
                    docs.append(Document(page_content=doc["page_content"],metadata={ "source":doc["source"]}))
                    if len(docs) > max_doc_size:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=params.chunk_size, chunk_overlap=params.chunk_overlap)
                        split_docs = text_splitter.split_documents(docs) 
                        print(f"Build vector db in {self.db_dir}. total docs: {len(docs)} total split docs: {len(split_docs)}")
                        db = FAISS.from_documents(split_docs, self.embeddings)
                        counter += 1
                        temp_path = os.path.join(self.db_dir,str(counter))
                        paths.append(temp_path)
                        db.save_local(temp_path)
                        docs.clear()

        if  len(docs) > 0:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=params.chunk_size, chunk_overlap=params.chunk_overlap)
            split_docs = text_splitter.split_documents(docs) 
            print(f"Build vector db in {self.db_dir}. total docs: {len(docs)} total split docs: {len(split_docs)}")
            db = FAISS.from_documents(split_docs, self.embeddings)
            counter += 1
            temp_path = os.path.join(self.db_dir,str(counter))
            paths.append(temp_path)
            db.save_local(temp_path)
            docs.clear()
        
        print(f"load vector index from {paths[0]}")
        t_db = FAISS.load_local(paths[0],self.embeddings)
        for item in paths[1:]:
            print(f"merge vector index from {paths[0]}")
            t_db.merge_from(FAISS.load_local(item,self.embeddings))
        t_db.save_local(self.db_dir)
        
        print(F"clean temp file...")
        for p in paths:
            shutil.rmtree(p)