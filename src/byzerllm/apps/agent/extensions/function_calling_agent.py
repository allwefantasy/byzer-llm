from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,code_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
from .. import get_agent_name,run_agent_func,ChatResponse
from ....utils import generate_str_md5
from byzerllm.utils.client import LLMHistoryItem,LLMRequest
import uuid
import json
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter,Document
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    
class FunctionCallingAgent(ConversableAgent): 
    DEFAULT_SYSTEM_MESSAGE='''You are a helpful assistant with access to the following functions:

{serialize_function_to_json(get_weather)}

{serialize_function_to_json(calculate_mortgage_payment)}

{serialize_function_to_json(get_directions)}

{serialize_function_to_json(get_article_details)}

To use these functions respond with:
<multiplefunctions>
    <functioncall> {fn} </functioncall>
    <functioncall> {fn} </functioncall>
    ...
</multiplefunctions>

Edge cases you must handle:
- If there are no functions that match the user request, you will respond politely that you cannot help.'''

    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval,        
        code_agent: Union[Agent, ClientActorHandle,str],  
        sql_reviewer_agent: Union[Agent, ClientActorHandle,str],      
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

        self.code_agent = code_agent
        self.sql_reviewer_agent = sql_reviewer_agent
        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)           
        self.register_reply([Agent, ClientActorHandle,str], SparkSQLAgent.generate_sql_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 
    
    
    def generate_sql_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
        ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  

        if get_agent_name(sender) == get_agent_name(self.sql_reviewer_agent):
            return False,None
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)]

        # give the response to    
        _,v = self.generate_llm_reply(raw_message,messages,sender)
        codes = code_utils.extract_code(v)
        has_sql_code = False

        for code in codes:                  
            if code[0]!="unknown":                
                has_sql_code = True           

        if has_sql_code:                
            self.send(messages[-1],self.sql_reviewer_agent,request_reply=False)
            self.send({
                    "content":v
                },self.sql_reviewer_agent)
            
            reply = self.chat_messages[get_agent_name(self.sql_reviewer_agent)][-2]["content"]
            # reply = run_agent_func(self.sql_reviewer_agent,"get_chat_messages")[get_agent_name(self.sql_reviewer_agent)][-2]        
            return True, {"content":reply,"metadata":{"TERMINATE":True}}             
        
        return True,  {"content":v,"metadata":{"TERMINATE":True}}
            
        
        



            

        