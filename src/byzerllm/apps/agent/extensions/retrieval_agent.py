from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
from .. import get_agent_name,run_agent_func,ChatResponse
from ....utils import generate_str_md5
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



class RetrievalAgent(ConversableAgent): 
    PROMPT_DEFAULT = """You're a retrieve augmented chatbot. You answer user's questions based on your own knowledge and the
context provided by the user. You should follow the following steps to answer a question:
Step 1, you estimate the user's intent based on the question and context. The intent can be a code generation task or
a question answering task.
Step 2, you reply based on the intent.
If you can't answer the question with or without the current context, you should reply exactly `UPDATE CONTEXT`.
If user's intent is code generation, you must obey the following rules:
Rule 1. You MUST NOT install any packages because all the packages needed are already installed.
Rule 2. You must follow the formats below to write your code:
```language
# your code
```

If user's intent is question answering, you must give as short an answer as possible.

User's question is: {input_question}

Context is: {input_context}
"""

    PROMPT_CODE = """You're a retrieve augmented coding assistant. You answer user's questions based on your own knowledge and the
    context provided by the user.
    If you can't answer the question with or without the current context, you should reply exactly `UPDATE CONTEXT`.
    For code generation, you must obey the following rules:
    Rule 1. You MUST NOT install any packages because all the packages needed are already installed.
    Rule 2. You must follow the formats below to write your code:
    ```language
    # your code
    ```

    User's question is: {input_question}

    Context is: {input_context}
    """

    PROMPT_QA = """You're a retrieve augmented chatbot. You answer user's questions based on your own knowledge and the
    context provided by the user.
    If you can't answer the question with or without the current context, you should reply exactly `UPDATE CONTEXT`.
    You must give as short an answer as possible.
    """   

    DEFAULT_SYSTEM_MESSAGE = PROMPT_QA
    
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval,        
        code_agent: Union[Agent, ClientActorHandle,str],
        byzer_engine_url: str="http://127.0.0.1:9003/model/predict",        
        retrieval_cluster:str="data_analysis",
        retrieval_db:str="data_analysis",
        update_context_retry: int = 3,
        chunk_size_in_context: int = 1,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,        
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[Dict, bool]] = False,
        **kwargs,
    ):       
        super().__init__(
            name,
            llm,retrieval,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            code_execution_config=code_execution_config,            
            **kwargs,
        )
        self.chat_name = name
        self.owner = name
        self.code_agent = code_agent

        self.byzer_engine_url = byzer_engine_url

        self.update_context_retry = update_context_retry
        self.chunk_size_in_context = chunk_size_in_context

        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], RetrievalAgent.generate_retrieval_based_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 
                
        self.retrieval_cluster = retrieval_cluster
        self.retrieval_db = retrieval_db          
        
                
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
field(content_vector,array(float))
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
field(chunk_vector,array(float))
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

    def generate_retrieval_based_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)]
                
        new_message = messages[-1]

        if "file_path" in new_message["metadata"]:
            file_path = new_message["metadata"]["file_path"]
            content = open(file_path,"r").read()
            self.save_text_content(owner=self.owner,title="",content=content,url=file_path) 

        if "file_ref" in new_message["metadata"]:
            file_ref = new_message["metadata"]["file_ref"]
            file_path = new_message["metadata"].get("file_path","")
            content = ray.get(file_ref)
            self.save_text_content(owner=self.owner,title="",content=content,url=file_path) 
        
        contents = self.search_content_chunks(owner=self.owner,q=new_message["content"],limit=100,return_json=False)
        # split the contents into n groups
        groups = []
        
        if not contents:
            return True,"FAIL TO ANSWER TERMINATE"
        
        for i in range(0,len(contents),self.chunk_size_in_context):
            groups.append(contents[i:i + self.chunk_size_in_context])

    
        current_doc = 0        

        input_context = json.dumps([{"content":x["raw_chunk"]} for x in groups[current_doc]],ensure_ascii=False,indent=4)

        prompt = PromptTemplate.from_template('''User's question is: {input_question}

Context is: 

```json                                                                                            
{input_context}
```
''').format(input_question=new_message["content"],input_context=input_context)
        
        new_message = {"content":prompt,"role":"user"}                    
        final,v = self.generate_llm_reply(None,[new_message],sender)
                
        update_context_case = "UPDATE CONTEXT" in v[-20:].upper() or "UPDATE CONTEXT" in v[:20].upper()
        while update_context_case:            
            current_doc += 1
            if current_doc >= self.update_context_retry or current_doc >= len(contents):
                break

            if not groups[current_doc] :
                break
            
            input_context = json.dumps([{"content":x["raw_chunk"]} for x in groups[current_doc]],ensure_ascii=False,indent=4)

            prompt = PromptTemplate.from_template('''User's question is: {input_question}

Context is: 

```json                                                                                            
{input_context}
```
''').format(input_question=new_message["content"],input_context=input_context)
            new_message = {"content":prompt,"role":"user"}
            final,v = self.generate_llm_reply(None,[new_message],sender)
            update_context_case = "UPDATE CONTEXT" in v[-20:].upper() or "UPDATE CONTEXT" in v[:20].upper()                

        if update_context_case:
            return True,{"content":"FAIL TO ANSWER","metadata":{"TERMINATE":True}}
        else:
            return True, {"content":v,"metadata":{"TERMINATE":True}}
                
        
        
        
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


    def save_text_content(self,owner:str,title:str,content:str,url:str,auth_tag:str=""):

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
            "content_vector":self.emb(content[0:10000])
            }]
        self.retrieval.build_from_dicts(self.retrieval_cluster,self.retrieval_db,"text_content",text_content)
        
        content_chunks= self.split_text_into_chunks(content)
        
        
        text_content_chunks = [{"_id":f'''{text_content[0]["_id"]}_{i}''',
            "doc_id":text_content[0]["_id"],
            "owner":owner,
            "chunk":self.search_tokenize(item),
            "raw_chunk":item,
            "chunk_vector":self.emb(item)
            } for i,item in enumerate(content_chunks)]
        
        self.retrieval.build_from_dicts(self.retrieval_cluster,self.retrieval_db,"text_content_chunk",text_content_chunks)    
    
    def _owner_filter(self,owner:str):
        return {"field":"owner","value":owner}
            
    def search_content_chunks(self,q:str,owner:str,limit:int=4,return_json:bool=True):   
        docs = self.retrieval.search(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content_chunk",
                                         filters={"and":[self._owner_filter(owner)]},
                                        keyword=self.search_tokenize(q),fields=["chunk"],
                                        vector=self.emb(q),vectorField="chunk_vector",
                                        limit=limit)])

        if return_json:
            context = json.dumps([{"content":x["raw_chunk"]} for x in docs],ensure_ascii=False,indent=4)    
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
                                     filters={"and":[self._owner_filter(owner=owner)]},
                                    keyword=chat_name,fields=["chat_name"],
                                    vector=self.emb(q),vectorField="content_vector",
                                    limit=1000)])
        docs = [doc for doc in docs if doc["role"] == "user" and doc["chat_name"] == chat_name]
        if return_json:
            context = json.dumps([{"content":x["raw_chunk"]} for x in docs[0:limit]],ensure_ascii=False,indent=4)    
            return context 
        else:
            return docs[0:limit]    

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