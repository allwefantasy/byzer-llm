
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
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter,Document
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

import jieba  

class SimpleRetrieval:
    def __init__(self,llm:ByzerLLM, retrieval: ByzerRetrieval,
                 retrieval_cluster:str="byzerai_store",
                 retrieval_db:str="byzerai_store",
                 max_output_length=10000):
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
field(owner,string),
field(content,string,analyze),
field(json_data,string),
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
field(owner,string),
field(chunk,string,analyze),
field(raw_chunk,string),
field(chunk_vector,array(float)),
field(metadata,string),
field(created_time,long,sort)
)''',
                location=f"/tmp/{self.retrieval_cluster}",num_shards=1                
           )) 
           if not self.retrieval.check_table_exists(self.retrieval_cluster,self.retrieval_db,"user_memory"):
                self.retrieval.create_table(self.retrieval_cluster,tableSettings=TableSettings(
                        database=self.retrieval_db,
                        table="user_memory",schema='''st(
        field(_id,string),
        field(chat_name,string),
        field(role,string),
        field(owner,string),
        field(content,string,analyze),
        field(raw_content,string),
        field(auth_tag,string,analyze),
        field(created_time,long,sort),
        field(chat_name_vector,array(float)),
        field(content_vector,array(float))
        )
        ''',
                        location=f"/tmp/{self.retrieval_cluster}",num_shards=1
                ))  

    
    
    def get_conversations_as_history(self,owner:str,chat_name:str,limit=1000)->List[LLMHistoryItem]:
        chat_history = self.get_conversations(owner,chat_name,limit=limit)        
        chat_history = [LLMHistoryItem(item["role"],item["raw_content"]) for item in chat_history]
        return chat_history    


    def save_doc(self,data:List[Dict[str,Any]],owner:Optional[str]=None,):

        if not self.retrieval:
            raise Exception("retrieval is not setup")
  
        owner = owner or "default"
    
        result = []
        for item in data:
            doc_id = item["doc_id"]
            collection = item["collection"]
            _id = f'{collection}/{doc_id}',
            result.append({"_id":_id,
            "doc_id":doc_id,               
            "json_data":item["json_data"], 
            "collection":collection,            
            "content":self.search_tokenize(item["content"]),
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
        filters = {}
    
        if query_str is not None:
            keyword = self.search_tokenize(query_str)
            fields = ["content"]

        if query_embedding is not None:
            vector = query_embedding
            vectorField = "content_vector"    
        
        
        if doc_ids is not None:
            filters = {"or":[{"field":"doc_id","value":x} for x in doc_ids]}
        
        docs = self.retrieval.search(self.retrieval_cluster,
                        [SearchQuery(self.retrieval_db,"text_content_chunk",
                                    filters=filters,
                                    keyword=keyword,fields=fields,
                                    vector=vector,vectorField=vectorField,
                                    limit=limit)])
    
        if return_json:
            context = json.dumps([{"content":x["raw_chunk"]} for x in docs],ensure_ascii=False,indent=4)    
            return context 
        else:
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
        docs = self.retrieval.search(self.retrieval_cluster,
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
            
            text_content_chunks.append({"_id":chunk_id,
                "doc_id":ref_doc_id or "",
                "owner":owner or "default",
                "chunk":self.search_tokenize(chunk_content),
                "raw_chunk":chunk_content,
                "chunk_vector":chunk_embedding,                
                "metadata":json.dumps(metadata,ensure_ascii=False),
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
    
    def delete_doc(self,doc_ids:List[str],collection:str):
        ids = []
        for doc_id in doc_ids:
            ids.append(f'{collection}/{doc_id}')
        self.retrieval.delete_by_ids(self.retrieval_cluster,self.retrieval_db,"text_content",[ids])

    def commit_doc(self):
        self.retrieval.commit(self.retrieval_cluster,self.retrieval_db)    
                    
        
    def search_memory(self,chat_name:str, owner:str, q:str,limit:int=4,return_json:bool=True):
        docs = self.retrieval.search(self.retrieval_cluster,
                        [SearchQuery(self.retrieval_db,"user_memory",
                                    filters={"and":[self._owner_filter(owner=owner),{"field":"chat_name","value":chat_name}]},
                                    sorts =[{"created_time":"desc"}],
                                    keyword=self.search_tokenize(q),fields=["content"],
                                    vector=self.emb(q),vectorField="content_vector",
                                    limit=1000)])
        docs = [doc for doc in docs if doc["role"] == "user" and doc["chat_name"] == chat_name]
        if return_json:
            context = json.dumps([{"content":x["raw_chunk"]} for x in docs[0:limit]],ensure_ascii=False,indent=4)    
            return context 
        else:
            return docs[0:limit]    

    def emb(self,s:str):        
        return self.llm.emb(self.llm.default_emb_model_name,LLMRequest(instruction=s))[0].output[0:1024] 


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
    
    def save_conversation(self,owner:str,chat_name:str,role:str,content:str):
        if not self.retrieval:
            raise Exception("retrieval is not setup")                                

        if chat_name is None:
            chat_name = content[0:10]   

        if len(content) > self.max_output_length:
            raise Exception(f"The response content length {len(content)} is larger than max_output_length {self.max_output_length}")

        data = [{"_id":str(uuid.uuid4()),
                "chat_name":chat_name,
                "role":role,
                "owner":owner,
                "content":self.search_tokenize(content),
                "raw_content":content,
                "auth_tag":"",
                "created_time":int(time.time()*1000),
                "chat_name_vector":self.emb(chat_name),
                "content_vector":self.emb(content)}]    

        self.retrieval.build_from_dicts(self.retrieval_cluster,self.retrieval_db,"user_memory",data)

    def get_conversations(self,owner:str, chat_name:str,limit=1000)->List[Dict[str,Any]]:
        docs = self.retrieval.filter(self.retrieval_cluster,
                        [SearchQuery(self.retrieval_db,"user_memory",
                                     filters={"and":[self._owner_filter(owner),{"field":"chat_name","value":chat_name}]},
                                     sorts=[{"created_time":"desc"}],
                                    keyword=None,fields=["chat_name"],
                                    vector=[],vectorField=None,
                                    limit=limit)])
        sorted_docs = sorted(docs[0:limit],key=lambda x:x["created_time"],reverse=False)
        return sorted_docs
    
               
    

