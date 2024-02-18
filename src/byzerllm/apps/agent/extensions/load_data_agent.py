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
from byzerllm.apps.llama_index import get_service_context,get_storage_context
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SentenceSplitter
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SentenceSplitter,SentenceWindowNodeParser
from llama_index import Document

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

import jieba     

class LoadDataAgent(ConversableAgent):      

    DEFAULT_SYSTEM_MESSAGE = "You're a retrieve augmented chatbot."
    
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
        

        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], LoadDataAgent.generate_load_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply)                       

        self.service_context = get_service_context(llm)
        self.storage_context = get_storage_context(llm,retrieval)
        
        
                        

    def generate_load_reply(
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
        documents = SimpleDirectoryReader(content).load_data()

        sp = SentenceSplitter(chunk_size=1024, chunk_overlap=0)
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",  
            sentence_splitter=sp  
        )

        nodes = node_parser.get_nodes_from_documents(
            documents, show_progress=True
        )

        _ = VectorStoreIndex(nodes=nodes, storage_context=self.storage_context, service_context=self.service_context)

        return True, {
            "content":"Data loaded successfully.",
            "metadata":{"agent":self.name,"TERMINATE":True}
        }
        
                
        
                    