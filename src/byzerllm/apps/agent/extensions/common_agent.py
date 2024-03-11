from .conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ...utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from .agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
from . import get_agent_name,run_agent_func,ChatResponse


class CommonAgent(ConversableAgent):    

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Based on the conversation and try you best to answer the user's quesion.
"""

    def __init__(
        self,
        name: str,
        llm: ByzerLLM,
        retrieval: ByzerRetrieval,
        chat_name:str,
        owner:str,
        code_agent: Union[Agent, ClientActorHandle,str],
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
        self._reply_func_list = []
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)           
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 
        
        