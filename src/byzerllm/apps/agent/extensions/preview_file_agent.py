from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
from .. import get_agent_name,run_agent_func,ChatResponse


class PreviewFileAgent(ConversableAgent):    

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
I have a file where the path is {file_path}, I want to use pandas to read it.The packages all are installed, you can use it directly.
Try to help me to generate python code which should match the following requirements:
1. try to read the file according the suffix of file name in Try block
2. if read success, set variable loaded_successfully to True, otherwise set it to False.
3. if loaded_successfully is True, then assigh the loaded data with head() to file_preview, otherwise assign error message to file_preview
4. make sure the loaded_successfully, file_preview are defined in the global scope
"""

    def __init__(
        self,
        name: str,
        llm: ByzerLLM,
        retrieval: ByzerRetrieval,
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
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], PreviewFileAgent.generate_code_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply)      
    
    def generate_code_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)]
        
        # if the message is not from the code agent, then generate code 
        # and talk to the code agent until the code agent gives the success message
        if get_agent_name(sender) != get_agent_name(self.code_agent):
   
            final,output = self.generate_llm_reply(raw_message,messages,sender)            
            # ask the code agent to execute the code             
            self.send(message=output,recipient=self.code_agent)

            # summarize the conversation so far  
            code_agent_messages = self._messages[get_agent_name(self.code_agent)]
            answer = code_agent_messages[-1]["content"] # self.generate_llm_reply(None,,sender)
            # give the result to the user             
            return True, answer + "\nTERMINATE"
        

        ## no code block found so the code agent return None
        if raw_message is None or isinstance(raw_message,str):
            return False, None
                
        raw_message: ChatResponse = raw_message
        if raw_message.status == 0:
            # stop the conversation if the code agent gives the success message
            return True, None
        else:
            # the code may be wrong, so generate a new code according to the conversation so far
            final,output = self.generate_llm_reply(raw_message,messages,sender)
            return True, output
        