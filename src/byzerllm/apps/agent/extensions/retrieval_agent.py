from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from .. import get_agent_name,run_agent_func,ChatResponse
from byzerllm.apps.agent.extensions.simple_retrieval_client import SimpleRetrievalClient
import uuid
import json
from langchain import PromptTemplate

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
        chat_name:str,
        owner:str,        
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
        self.chat_name = chat_name
        self.owner = owner
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

        self.simple_retrieval_client = SimpleRetrievalClient(llm=self.llm,
                                                             retrieval=self.retrieval,
                                                             retrieval_cluster=self.retrieval_cluster,
                                                             retrieval_db=self.retrieval_db,
                                                             )         
        
                        

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
            self.simple_retrieval_client.save_text_content(owner=self.owner,title="",content=content,url=file_path) 

        if "file_ref" in new_message["metadata"]:
            file_ref = new_message["metadata"]["file_ref"]
            file_path = new_message["metadata"].get("file_path","")
            content = ray.get(file_ref)
            self.simple_retrieval_client.save_text_content(owner=self.owner,title="",content=content,url=file_path) 
        
        top_k = 4
        if "top_k" in new_message["metadata"]:
            top_k = new_message["metadata"]["top_k"]    
        
        contents = self.simple_retrieval_client.search_content_chunks(owner=self.owner,q=new_message["content"],limit=top_k,return_json=False)
        for item in contents:
            temp = self.simple_retrieval_client.get_doc(item["doc_id"],owner=self.owner)
            item["doc_url"] = temp["url"]                        

        input_context = json.dumps([{"content":x["raw_chunk"]} for x in contents],ensure_ascii=False,indent=4)

        prompt = PromptTemplate.from_template('''User's question is: {input_question}

Context is: 

```json                                                                                            
{input_context}
```
''').format(input_question=new_message["content"],input_context=input_context)
        
        new_message = {"content":prompt,"role":"user"}
        id = str(uuid.uuid4())        
        v = self.llm.stream_chat_oai(conversations=self._system_message + [new_message])
        
        return self.stream_reply(v,contexts=contents)          
        
                
        
                    