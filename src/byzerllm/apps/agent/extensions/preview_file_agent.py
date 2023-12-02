from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
from .. import get_agent_name,run_agent_func,ChatResponse,modify_last_message,modify_message_content
from langchain import PromptTemplate
import json


class PreviewFileAgent(ConversableAgent):    

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant. You will use your knowledge to help the user to preview the file.Try
to use Python and Pandas to read the file and show the first 5 rows of the file. The user will mention the file path in his/her question.
The packages all are installed, you can use it directly.
Try to generate python code which should match the following requirements:
1. try to read the file according the suffix of file name in Try block
2. if read success, set variable loaded_successfully to True, otherwise set it to False.
3. if loaded_successfully is True, then assigh the loaded data with head() to file_preview, otherwise assign error message to file_preview
4. make sure the loaded_successfully, file_preview are defined in the global scope
"""    
    
    DEFAULT_USER_MESSAGE = """
The file path is: {file_path}. Try to preview the file.
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
            
            new_message = messages[-1] 
            content = PromptTemplate.from_template(self.DEFUALT_USER_MESSAGE).format(file_path=new_message["metadata"]["file_path"])
            new_messages = modify_last_message(messages,modify_message_content(new_message,content))
            _,code = self.generate_llm_reply(raw_message,new_messages,sender)            
            # ask the code agent to execute the code  
            temp_message = {
                "content":code,
                "target_names":{"loaded_successfully":True,"file_preview":""}
            }           
            self.send(message=temp_message,recipient=self.code_agent)

            # summarize the conversation so far  
            code_agent_messages = self._messages[get_agent_name(self.code_agent)]
            
            response:ChatResponse = code_agent_messages[-1]["metadata"] # self.generate_llm_reply(None,,sender)            
            file_preview = response.variables["file_preview"].to_csv(index=False)    
            
            return True, file_preview + "\nTERMINATE"
        

        ## no code block found so the code agent return None
        if raw_message is None or isinstance(raw_message,str):
            return False, None
                
        raw_message: ChatResponse = raw_message

        if raw_message.status == 0 and "loaded_successfully" in raw_message.variables and raw_message.variables["loaded_successfully"]:
            # stop the conversation if the code agent gives the success message
            return True, None
        else:
            # the code may be wrong, so generate a new code according to the conversation so far
            final,output = self.generate_llm_reply(raw_message,messages,sender)
            return True, output
        