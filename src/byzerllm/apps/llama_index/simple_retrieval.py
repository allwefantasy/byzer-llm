
from byzerllm.utils.retrieval import ByzerRetrieval
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from byzerllm.utils.client import ByzerLLM
from byzerllm.utils import generate_str_md5
from byzerllm.utils.retrieval import ByzerRetrieval
import time
from byzerllm.utils.client import LLMHistoryItem,LLMRequest
from byzerllm.utils.retrieval import TableSettings,SearchQuery
import uuid
import json
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

import jieba  

class SimpleRetrieval:
    def __init__(self,llm:ByzerLLM, retrieval: ByzerRetrieval,
                 chunk_collection: Optional[str] = "default",
                 retrieval_cluster:str="byzerai_store",
                 retrieval_db:str="byzerai_store",
                 max_output_length=10000):
        self.chunk_collection = chunk_collection
        self.retrieval_cluster = retrieval_cluster
        self.retrieval_db = retrieval_db
        self.max_output_length = max_output_length
        self.llm = llm
        self.retrieval = retrieval                
        if self.llm.default_emb_model_name is None:
            raise Exception(f'''
emb model does not exist. Try to use `llm.setup_default_emb_model_name` to set the default emb model name.
''')        

        # create the retrieval database/table if not exists
        if self.retrieval and not self.retrieval.check_table_exists(self.retrieval_cluster,self.retrieval_db,"text_content"):
           self.retrieval.create_table(self.retrieval_cluster,tableSettings=TableSettings(
                database=self.retrieval_db,
                table="text_content",schema='''st(
field(_id,string),
field(doc_id,string),
field(file_path,string),
field(owner,string),
field(content,string,analyze),
field(raw_content,string,no_index),
field(metadata,string,analyze),
field(json_data,string,no_index),
field(collection,string),
field(created_time,long,sort)
)''',
                location=f"/tmp/{self.retrieval_cluster}",num_shards=1 
           ))

           self.retrieval.create_table(self.retrieval_cluster,tableSettings=TableSettings(
                database=self.retrieval_db,
                table="text_content_chunk",schema='''st(
field(_id,string),
field(doc_id,string),
field(file_path,string),
field(owner,string),
field(chunk,string,analyze),
field(raw_chunk,string,no_index),
field(chunk_vector,array(float)),
field(metadata,string,analyze),
field(json_data,string,no_index),
field(chunk_collection,string),
field(created_time,long,sort)
)''',
                location=f"/tmp/{self.retrieval_cluster}",num_shards=1                
           )) 
            

    def save_doc(self,data:List[Dict[str,Any]],owner:Optional[str]=None,):

        if not self.retrieval:
            raise Exception("retrieval is not setup")
  
        owner = owner or "default"
    
        result = []
        for item in data:
            
            doc_id = item["doc_id"]
            json_data_obj = json.loads(item["json_data"])
            collection = item["collection"]
            _id = f"{collection}/{doc_id}"
                    
            file_path = json_data_obj.get("metadata",{}).get("file_path","")                
            result.append({"_id":_id,
            "doc_id":doc_id,     
            "file_path":file_path,          
            "json_data":item["json_data"],
            "metadata":self.search_tokenize(item["json_data"]), 
            "collection":collection,            
            "content":self.search_tokenize(item["content"]),
            "raw_content":item["content"],
            "owner":owner,          
            "created_time":int(time.time()*1000),
            })        
                    
        self.retrieval.build_from_dicts(self.retrieval_cluster,self.retrieval_db,"text_content",result)            

    
    def _owner_filter(self,owner:str):
        return {"field":"owner","value":owner}
            
    def search_content_chunks(self,
                              owner:str,
                              query_str:Optional[str]=None,
                              query_embedding:Optional[List[float]]=None,  
                              doc_ids:Optional[List[str]]=None,                            
                              limit:int=4,        
                              return_json:bool=True):   
        keyword = None
        fields = []
        vector = []
        vectorField = None
        filters = {"and":[{"field":"chunk_collection","value":self.chunk_collection}]}
    
        if query_str is not None:
            keyword = self.search_tokenize(query_str)
            fields = ["chunk"]

        if query_embedding is not None:
            vector = query_embedding
            vectorField = "chunk_vector"    
        
        
        if doc_ids:
            filters["and"].append({"or":[{"field":"doc_id","value":x} for x in doc_ids]})
        
        query = SearchQuery(self.retrieval_db,"text_content_chunk",
                                    filters=filters,
                                    keyword=keyword,fields=fields,
                                    vector=vector,vectorField=vectorField,
                                    limit=limit)
        
        docs = self.retrieval.search(self.retrieval_cluster,
                        [query])
    
        if return_json:
            context = json.dumps([{"content":x["raw_chunk"]} for x in docs],ensure_ascii=False,indent=4)    
            return context 
        else:
            return docs
        

    def search_content_by_filename(self,filename:str,collection:str): 
        filters = [{"field":"chunk_collection","value":collection}]
        docs = self.retrieval.search(self.retrieval_cluster,
                        [SearchQuery(self.retrieval_db,"text_content_chunk",
                                    filters={"and":filters},
                                    keyword=self.search_tokenize(filename),fields=["metadata"],
                                    vector=[],vectorField=None,
                                    limit=10000)])
        return docs 

    def search_content(self,q:str,owner:str,url:str,auth_tag:str=None,limit:int=4,return_json:bool=True): 
        filters = [self._owner_filter(owner)]
        
        if auth_tag:
            filters.append({"field":"auth_tag","value":self.search_tokenize(auth_tag)})
        
        if url:
            filters.append({"field":"url","value":url})    

        if q:
            keyword = self.search_tokenize(q)
            vector = self.emb(q)
            vectorField = "content_vector"
            fields = ["content"]
        else:
            keyword = None
            vector = []
            vectorField = None
            fields = []

        docs = self.retrieval.search(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content",
                                         filters={"and":filters},
                                        keyword=keyword,fields=fields,
                                        vector=vector,vectorField=vectorField,
                                        limit=limit)])

        if return_json:
            context = json.dumps([{"content":x["raw_content"]} for x in docs],ensure_ascii=False,indent=4)    
            return context 
        else:
            return docs 

    def get_chunk_by_id(self,chunk_id:str):
        filters = {"and":[{"field":"_id","value":chunk_id}]}        
        docs = self.retrieval.filter(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content_chunk",
                                         filters=filters,
                                        keyword=None,fields=[],
                                        vector=[],vectorField=None,
                                        limit=1)])
        return docs
    
    def get_chunks_by_docid(self,doc_id:str):
        docs = self.retrieval.filter(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content_chunk",
                                         filters={"and":[{"field":"doc_id","value":doc_id}]},
                                        keyword=None,fields=[],
                                        vector=[],vectorField=None,
                                        limit=100000)])
        return docs
    
    def delete_chunk_by_ids(self,chunk_ids:List[str]):        
        self.retrieval.delete_by_ids(self.retrieval_cluster,self.retrieval_db,"text_content_chunk",chunk_ids)
        self.retrieval.commit(self.retrieval_cluster,self.retrieval_db)    
    
    def save_chunks(self,chunks:List[Dict[str,Any]]):        
        text_content_chunks = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            ref_doc_id = chunk["ref_doc_id"]
            owner = chunk.get("owner","default")
            chunk_content = chunk["chunk_content"]
            chunk_embedding = chunk["chunk_embedding"]
            metadata = chunk.get("metadata",{})
            _id = f"{self.chunk_collection}/{chunk_id}"
            file_path = metadata.get("file_path","")            
            
            text_content_chunks.append({"_id":_id,
                "doc_id":ref_doc_id or "",
                "file_path": file_path,
                "owner":owner or "default",                
                "chunk":self.search_tokenize(chunk_content),
                "raw_chunk":chunk_content,
                "chunk_vector":chunk_embedding,  
                "chunk_collection":self.chunk_collection,              
                "metadata":self.search_tokenize(json.dumps(metadata,ensure_ascii=False)),
                "json_data":json.dumps(metadata,ensure_ascii=False),
                "created_time":int(time.time()*1000),
                })                   
            
        self.retrieval.build_from_dicts(self.retrieval_cluster,
                                        self.retrieval_db,
                                        "text_content_chunk",text_content_chunks)           

            
        
    def get_doc(self,doc_id:str,collection:str):
        filters = {"and":[{"field":"_id","value":f'{collection}/{doc_id}'}]}        
        docs = self.retrieval.filter(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content",
                                         filters=filters,
                                        keyword=None,fields=[],
                                        vector=[],vectorField=None,
                                        limit=1)])
        return docs[0] if docs else None
    
    @DeprecationWarning
    def delete_doc(self,doc_ids:List[str],collection:str):
        ids = []
        for doc_id in doc_ids:
            ids.append(f'{collection}/{doc_id}')
        self.retrieval.delete_by_ids(self.retrieval_cluster,self.retrieval_db,"text_content",[ids])

    def truncate_table(self):
        self.retrieval.truncate(self.retrieval_cluster,self.retrieval_db,"text_content")
        self.retrieval.truncate(self.retrieval_cluster,self.retrieval_db,"text_content_chunk")
        self.commit_doc()
        self.commit_chunk()

    def delete_doc_and_chunks_by_filename(self,collection:str,file_name:str):
        doc = self.get_doc(f"{collection}/ref_doc_info",file_name)
                
        if doc:
            node_ids = json.loads(doc["metadata"])["node_ids"]
            for node_id in node_ids:
                id = f"{collection}/data/{node_id}"
                self.retrieval.delete_by_ids(self.retrieval_cluster,self.retrieval_db,"text_content",[id])

                id = f"{collection}/metadata/{node_id}"
                self.retrieval.delete_by_ids(self.retrieval_cluster,self.retrieval_db,"text_content",[id])

                ## default/index/ can not be deleted
            
            ##  cleanup the chunk collection  
            self.retrieval.delete_by_filter(self.retrieval_cluster,self.retrieval_db,"text_content_chunk",
                                            {"file_path":file_name})                      


    def delete_from_doc_collection(self,collection:str):
        for suffix in ["index","data","ref_doc_info","metadata"]:
            self.retrieval.delete_by_filter(self.retrieval_cluster,self.retrieval_db,"text_content",
                                            {"collection":f"{collection}/{suffix}"})
        

    def delete_from_chunk_collection(self,collection:str):
        self.retrieval.delete_by_filter(self.retrieval_cluster,self.retrieval_db,"text_content_chunk",
                                        {"chunk_collection":collection})    

    def commit_doc(self):
        self.retrieval.commit(self.retrieval_cluster,self.retrieval_db,"text_content")    

    def commit_chunk(self):
        self.retrieval.commit(self.retrieval_cluster,self.retrieval_db,"text_content_chunk")    
                                  

    def emb(self,s:str):        
        return self.llm.emb(self.llm.default_emb_model_name,LLMRequest(instruction=s))[0].output


    def split_text_into_chunks(self,s:str):
        # self.llm.apply_sql_func(
        #     '''select llm_split(value,array(",","。","\n"),1600) as value ''',[{"value":content}],
        #     url=self.byzer_engine_url
        #     )["value"]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        split_docs = text_splitter.split_documents([Document(page_content=s)])         
        return [s.page_content for s in split_docs] 

    
    def search_tokenize(self,s:str):        
        seg_list = jieba.cut(s, cut_all=False)
        # return self.llm.apply_sql_func("select mkString(' ',parse(value)) as value",[
        # {"value":s}],url=self.byzer_engine_url)["value"]
        return " ".join(seg_list) 
        
    
               
    

