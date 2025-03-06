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
from byzerllm.apps.llama_index import get_service_context,get_storage_context
from llama_index import VectorStoreIndex
from llama_index.query_engine import SubQuestionQueryEngine


try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

from llama_index.tools import QueryEngineTool, ToolMetadata    



class LlamaIndexSubQuestionAgent(ConversableAgent): 
    PROMPT_DEFAULT = """You're a retrieve augmented chatbot. """    
    DEFAULT_SYSTEM_MESSAGE = PROMPT_DEFAULT
    
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval,        
        chat_name:str,
        owner:str,                
        update_context_retry: int = 3,        
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
        self.update_context_retry = update_context_retry


        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], LlamaIndexSubQuestionAgent.generate_retrieval_based_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 
        self.service_context = get_service_context(llm)
        self.storage_context = get_storage_context(llm,retrieval)        
              
        
                        

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
        index = VectorStoreIndex.from_vector_store(vector_store = self.storage_context.vector_store,service_context=self.service_context)
        vector_query_engine = index.as_query_engine()
        query_engine_tools = [
                                QueryEngineTool(
                                    query_engine=vector_query_engine,
                                    metadata=ToolMetadata(
                                        name="common",
                                        description="common",
                                    ),
                                                ),
                            ]                

        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            service_context=self.service_context,
            use_async=True,
        )
        response = query_engine.query(new_message["content"])        
        return True, {
            "content":response.response,
            "metadata":{"agent":self.name,"TERMINATE":True}
        }
        
                
        
                    