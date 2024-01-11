from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,message_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from .. import get_agent_name,run_agent_func,ChatResponse
from byzerllm.apps.agent.extensions.simple_retrieval_client import SimpleRetrievalClient
import uuid
import json
from langchain import PromptTemplate
   



class RhetoricalAgent(ConversableAgent): 
    
    DEFAULT_SYSTEM_MESSAGE = '''你是一个非常善于反思和总结的AI助手。'''
    
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval,     
        chat_name:str,
        owner:str,           
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
        
        self.update_context_retry = update_context_retry
        self.chunk_size_in_context = chunk_size_in_context

        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], RhetoricalAgent.generate_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 
                
        self.retrieval_cluster = retrieval_cluster
        self.retrieval_db = retrieval_db 

        self.simple_retrieval_client = SimpleRetrievalClient(llm=self.llm,
                                                             retrieval=self.retrieval,
                                                             retrieval_cluster=self.retrieval_cluster,
                                                             retrieval_db=self.retrieval_db,
                                                             )         
        

    def is_update_context(self,v:str):
        update_context_case = "UPDATE CONTEXT" in v[-20:].upper() or "UPDATE CONTEXT" in v[:20].upper()                                    
        return update_context_case

    def generate_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  

        if messages is None:
            messages = self._messages[get_agent_name(sender)]  

        m = messages[-1]
        
        old_conversations = self.simple_retrieval_client.search_content(q=m["content"],owner=self.owner,url="rhetorical",limit=3)
        if len(old_conversations) != 0:
            c = json.dumps(old_conversations,ensure_ascii=False)
            self.update_system_message(f'''{self.DEFAULT_SYSTEM_MESSAGE}\n下面是我们以前对话的内容总结:
```json
{c}                                       
```  
你在回答我的问题的时候，可以参考这些内容。''')
                         
        last_conversation = [{"role":"user","content":'''回顾前面我们对话，我提出了很多问题，但是有些问题你没办法直接回答，让我补充了一些信息，然后你才能回答我的问题。
找到这些内容，并且做个总结'''}]
                
        _,v2 = self.generate_llm_reply(raw_message,message_utils.padding_messages_merge(old_conversations + messages + last_conversation),sender)
        self.simple_retrieval_client.save_text_content(owner=self.owner,title="",content=v2,url="rhetorical",auto_chunking=False)
        return True, None 
                
        
                    