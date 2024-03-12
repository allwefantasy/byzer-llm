from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from .. import get_agent_name,run_agent_func,ChatResponse
from byzerllm.apps.llama_index.simple_retrieval import SimpleRetrieval
import uuid
import json
from langchain import PromptTemplate
from byzerllm.apps.llama_index import get_service_context,get_storage_context
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SentenceSplitter
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SentenceSplitter,SentenceWindowNodeParser
from llama_index import (  
    Document,  
    get_response_synthesizer,
)
from llama_index.indices.document_summary import DocumentSummaryIndex
import pydantic

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

import jieba  

class AgentData(pydantic.BaseModel):
    path:str = pydantic.Field(...,description="用户提及的路径全名")
    namespace:str = pydantic.Field(...,description="用户提及的命名空间名字,如果没有显示提及，默认为 default")
    index_mode:str = pydantic.Field(...,description="用户提及的索引模式，当用户说简单对应的值为 simple，当用户说复杂模式时，对应的值为complex")

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
        self.register_reply([Agent, ClientActorHandle,str], LoadDataAgent.generate_load_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply)         

        self.llm = llm
        self.retrieval = retrieval
                                    

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

        @self.llm.response()
        def extract_data(s:str)->AgentData:
            pass

        agent_data = extract_data(content)
        id = str(uuid.uuid4())        

        if not agent_data.path:
            def gen():            
                yield ("Please provide a valid path.",None)                               
            return self.stream_reply(gen(),contexts=[])           

        if not agent_data.namespace:
            agent_data.namespace = "default"     

        print(agent_data,flush=True)           

        service_context = get_service_context(self.llm)
        storage_context = get_storage_context(self.llm,self.retrieval,chunk_collection=agent_data.namespace,namespace=agent_data.namespace)

        retrieval_client = SimpleRetrieval(llm=self.llm,retrieval=self.retrieval)
        retrieval_client.delete_from_doc_collection(agent_data.namespace)
        retrieval_client.delete_from_chunk_collection(agent_data.namespace)

        documents = SimpleDirectoryReader(agent_data.path).load_data()

        sp = SentenceSplitter(chunk_size=1024, chunk_overlap=0)        

        nodes = sp.get_nodes_from_documents(
            documents, show_progress=True
        )
        _ = VectorStoreIndex(nodes=nodes, storage_context=storage_context, service_context=service_context)
        
        def gen():                    
            yield (f"Data loaded successfully with mode {agent_data.index_mode}.",None)        
        self.put_stream_reply(id,gen())
        return True, {
            "content":id,
            "metadata":{"agent":self.name,"TERMINATE":True,"stream":True,"stream_id":id,"contexts":[]}
        }                 
        
                
        
                    