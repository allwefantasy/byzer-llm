from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Generator
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


try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

import jieba     
import pydantic

class AgentData(pydantic.BaseModel):        
    namespace:str = pydantic.Field(...,description="用户提及的命名空间名字,如果没有提及，则设置为 default")        


class LlamaIndexRetrievalAgent(ConversableAgent): 
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
        self.register_reply([Agent, ClientActorHandle,str], LlamaIndexRetrievalAgent.generate_retrieval_based_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 
        self.llm = llm
        self.retrieval = retrieval                                                         

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
        content = new_message["content"]
        
        # @self.llm.response()
        # def extract_data(s:str)->AgentData:
        #     pass

        # agent_data = extract_data(content)
        agent_data = AgentData(namespace="default")
        if not agent_data.namespace:
            agent_data.namespace = "default" 

        print(f"{agent_data}",flush=True)    

        service_context = get_service_context(self.llm)
        storage_context = get_storage_context(self.llm,self.retrieval,chunk_collection=agent_data.namespace,namespace=agent_data.namespace)                

        index = VectorStoreIndex.from_vector_store(vector_store = storage_context.vector_store,service_context=service_context)
        query_engine = index.as_query_engine(streaming=True)        
        id = str(uuid.uuid4())        
        streaming_response = query_engine.query(content)
        
        def gen(): 
            t = ""           
            for response in streaming_response.response_gen:
                t += response
                yield (t,None)

        return self.stream_reply(gen(),contexts=[])  
        
        
                
        
                    