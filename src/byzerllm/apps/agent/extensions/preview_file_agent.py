from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,message_utils
from byzerllm.utils.retrieval import ByzerRetrieval
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
We have a file, the path is: {file_path}. Try to  write code preview this file. Make sure the {file_path} is defined in the code. We need to 
execute the code to preview the file. If the code is correct, the file will be loaded successfully and the first 5 rows of the file will be shown.
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
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], PreviewFileAgent.generate_code_reply) 
        self.register_reply([Agent, ClientActorHandle,str], PreviewFileAgent.reply_python_code_agent) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 

    def reply_python_code_agent(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]: 
        if get_agent_name(sender) != get_agent_name(self.code_agent):
            return False, None
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)]

        message = messages[-1]
        raw_message = messages[-1]["metadata"]["execute_result"]

        if raw_message.status == 0 and "loaded_successfully" in raw_message.variables and raw_message.variables["loaded_successfully"]:                                 
            # stop the conversation if the code agent gives the success message
            return True, None
        else:
            print(raw_message,flush=True)
            if message_utils.check_error_count(message,3):
                return True, {
                    "content":f'''Fail to load the file: {raw_message.variables.get("file_path","")}. reason: {raw_message.output}''' + "\nTERMINATE",
                    "metadata":{"TERMINATE":True,"code":1}
                }
            # the code may be wrong, so generate a new code according to the conversation so far 
            extra_messages = []            
            if "loaded_successfully" not in raw_message.variables:                
                extra_messages.append(self.create_temp_message("loaded_successfully is not defined"))
            
            elif raw_message.variables["loaded_successfully"] is False:                            
                extra_messages.append(self.create_temp_message("loaded_successfully is False, it means the file is not loaded successfully, check the file path and the code then try again"))
                

            _,code = self.generate_llm_reply(raw_message,messages + extra_messages,sender)
            m = self.create_temp_message(code)
            message_utils.copy_error_count(message,m)

            return True, message_utils.inc_error_count(m)
        
        
    
    def generate_code_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)]
        
        new_message = messages[-1] 
        file_path = new_message["metadata"]["file_path"]
        content = PromptTemplate.from_template(self.DEFAULT_USER_MESSAGE).format(file_path=new_message["metadata"]["file_path"])
        new_messages = modify_last_message(messages,modify_message_content(new_message,content))
        
        _,code = self.generate_llm_reply(raw_message,new_messages,sender)            
        
        # only the first time we should keep the message sent to the code agent which have file_path, file_ref
        # in message metadata, when the sandbox is created, then we will reuse the sandbox, no need to contain
        # the file_path, file_ref in the message metadata.         
        self.send(message=self.create_temp_message(code,new_message),recipient=self.code_agent)

        # get the code agent's reply
        last_message = self._messages[get_agent_name(self.code_agent)][-1]
        if last_message["metadata"].get("code",0) != 0:
            return True, {"content":f'''Fail to load the file: {file_path}. reason: {response.variables.get("file_preview","")}''',"metadata":{"TERMINATE":True,"code":1}}
        
        response:ChatResponse = last_message["metadata"]["execute_result"]

        if "loaded_successfully" not in response.variables or response.variables["loaded_successfully"] is False:
            return True, {"content":f'''Fail to load the file: {file_path}. reason: {response.variables.get("file_preview","")}''',"metadata":{"TERMINATE":True,"code":1}}
        
        file_preview = response.variables["file_preview"].to_csv(index=False)    
        
        return True, {"content":file_preview,"metadata":{"TERMINATE":True}}
            
            
        

        
        
    def create_temp_message(self,code,original_message=None):
        temp_message = {
            "content":code,
            "metadata":{
                "target_names":{"loaded_successfully":None,"file_preview":None}
            },
            "role":"user"
        }
        if original_message is not None:
            temp_message["metadata"] = {**original_message["metadata"],**temp_message["metadata"]}
        return temp_message    
        