
from byzerllm.utils.retrieval import ByzerRetrieval
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM
from ....utils import generate_str_md5
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

class SimpleRetrievalClient:
    def __init__(self,llm:ByzerLLM, retrieval: ByzerRetrieval,retrieval_cluster:str,retrieval_db:str,max_output_length=10000):
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
field(owner,string),
field(title,string,analyze),
field(content,string,analyze),
field(url,string),
field(raw_content,string),
field(auth_tag,string,analyze),
field(title_vector,array(float)),
field(content_vector,array(float)),
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
    
    def get_conversations_as_history(self,owner:str,chat_name:str,limit=1000)->List[LLMHistoryItem]:
        chat_history = self.get_conversations(owner,chat_name,limit=limit)        
        chat_history = [LLMHistoryItem(item["role"],item["raw_content"]) for item in chat_history]
        return chat_history    


    def save_text_content(self,owner:str,title:str,content:str,url:str,auth_tag:str="",auto_chunking:bool=True):

        if not self.retrieval:
            raise Exception("retrieval is not setup")

                        
        text_content = [{"_id":generate_str_md5(content),
            "title":self.search_tokenize(title),
            "content":self.search_tokenize(content[0:10000]),
            "owner":owner,
            "raw_content":content[0:10000],
            "url":url,
            "auth_tag":self.search_tokenize(auth_tag),
            "title_vector":self.emb(title),
            "content_vector":self.emb(content[0:2048]),
            "created_time":int(time.time()*1000),
            }]
        self.retrieval.build_from_dicts(self.retrieval_cluster,self.retrieval_db,"text_content",text_content)
        
        if auto_chunking:            
            content_chunks= self.split_text_into_chunks(content)
                        
            text_content_chunks = [{"_id":f'''{text_content[0]["_id"]}_{i}''',
                "doc_id":text_content[0]["_id"],
                "owner":owner,
                "chunk":self.search_tokenize(item),
                "raw_chunk":item,
                "chunk_vector":self.emb(item),
                "created_time":int(time.time()*1000),
                } for i,item in enumerate(content_chunks)]
            
            self.retrieval.build_from_dicts(self.retrieval_cluster,self.retrieval_db,"text_content_chunk",text_content_chunks)    

    
    def _owner_filter(self,owner:str):
        return {"field":"owner","value":owner}
            
    def search_content_chunks(self,q:str,owner:str,limit:int=4,return_json:bool=True):   
        docs = self.retrieval.search(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content_chunk",
                                         filters={"and":[self._owner_filter(owner)]},
                                        keyword=self.search_tokenize(q),fields=["content"],
                                        vector=self.emb(q),vectorField="content_vector",
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
        
    def get_doc(self,doc_id:str,owner:str):
        docs = self.retrieval.search(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content",
                                         filters={"and":[self._owner_filter(owner)]},
                                        keyword=doc_id,fields=["_id"],
                                        vector=[],vectorField=None,
                                        limit=1)])
        return docs[0] if docs else None
    
    def get_doc_by_url(self,url:str,owner:str):
        docs = self.retrieval.search(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content",
                                         filters={"and":[self._owner_filter(owner)]},
                                        keyword=url,fields=["url"],
                                        vector=[],vectorField=None,
                                        limit=1)])
        return docs[0] if docs else None
                
        
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
    
               
    

