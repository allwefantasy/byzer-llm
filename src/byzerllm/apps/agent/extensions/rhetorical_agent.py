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
   



class RhetoricalAgent(ConversableAgent): 
    
    DEFAULT_SYSTEM_MESSAGE = '''你只做两件事：
    1. 对我的提问，你有什么不理解的地方么？如果有，请不要生成代码，用中文询问我，并且给我可能的解决方案。
    2. 如果没有不理解的地方，请在最后直接回复 "TERMINATE"。
    '''
    
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval,                
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
        
        old_conversations = self.simple_retrieval_client.search_memory(chat_name=self.chat_name,owner=self.owner,q=m["content"])
        last_conversation = []
        # last_conversation = [{"role":"user","content":"首先先回答，你有什么不理解的地方么？如果有，请不要生成代码，用中文询问我，并且给我可能的解决方案。"}]

        _,v = self.generate_llm_reply(raw_message,old_conversations + messages + last_conversation,sender)
        return True, {"content":v}
                
        
                    