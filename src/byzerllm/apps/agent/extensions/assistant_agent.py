from .conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ...utils.client import ByzerLLM,message_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from .agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
from . import get_agent_name,run_agent_func,ChatResponse


class AssistantAgent(ConversableAgent):    

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.

注意：
1. 不要提供类似 if __name__ == '__main__' 的判断。
2. 代码需要通过 exec 函数执行，所以不要使用 return 语句。
3. 不要使用 input 这种需要用户输入的函数。
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
        self.register_reply([Agent, ClientActorHandle,str], AssistantAgent.generate_code_reply) 
        self.register_reply([Agent, ClientActorHandle,str], AssistantAgent.reply_python_code_agent) 
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
                    
        raw_message = messages[-1]["metadata"]["execute_result"]
        if raw_message.status == 0:
            # stop the conversation if the code agent gives the success message
            return True, None
        else:
            # the code may be wrong, so generate a new code according to the conversation so far
            if message_utils.check_error_count(messages[-1],3):
                return True, {
                    "content":f'''FAIL TO GNERATE CODE : {raw_message.output}''',
                    "metadata":{"TERMINATE":True,"code":1}
                }
            final,output = self.generate_llm_reply(raw_message,messages,sender)
            temp_message = {
                "content":output,
            }
            message_utils.copy_error_count(messages[-1],temp_message)
            message_utils.inc_error_count(temp_message)
            return True, temp_message

    def generate_code_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)]
        
        final,output = self.generate_llm_reply(raw_message,messages,sender)            
                                 
        self.send(message=output,recipient=self.code_agent)

        # summarize the conversation so far  
        last_message = self._messages[get_agent_name(self.code_agent)][-1]                
        if last_message["metadata"].get("code",0) != 0:
            return True, {"content":f'''FAIL TO GNERATE CODE ''',"metadata":{"TERMINATE":True,"code":1}}

        # give the result to the user             
        return True, {"content":last_message["content"],"metadata":{"TERMINATE":True}}
   
            
        

       
        