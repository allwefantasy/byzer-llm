from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,message_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
from .. import get_agent_name,run_agent_func,ChatResponse,modify_last_message,modify_message_content,count_messages_length
from langchain import PromptTemplate


class VisualizationAgent(ConversableAgent):  
    DEFAULT_SYSTEM_MESSAGE = '''You are a helpful AI assistant.
Solve visualization tasks using your coding and language skills.
You'll be asked to generate code to visualize data. You can only use python.
''' 

    DEFAULT_USER_MESSAGE = """
Please DO NOT consider the package installation, the packages all are installed, you can use it directly.

When the question require you to do visualization, please use package Plotly or matplotlib to do this.
Try to use base64 to encode the image, assign the base64 string to the variable named image_base64. 
Make sure the image_base64 defined in the global scope. 

Notice that ALWAYS create figure with `plt.figure()` before you plot the image.

Here is the example code how to save the plot to a BytesIO object and encode the image to base64:

```python
# Save the plot to a BytesIO object
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

# Encode the image to base64
image_base64 = base64.b64encode(buf.read()).decode('utf-8')
buf.close()
```

Please try to generate python code to analyze the file and answer the following questions:

{question}

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
        self.register_reply([Agent, ClientActorHandle,str], VisualizationAgent.generate_code_reply)
        self.register_reply([Agent, ClientActorHandle,str], VisualizationAgent.reply_python_code_agent) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 

    def create_temp_message(self,code,original_message=None):
        temp_message = {
            "content":code,
            "metadata":{
                "target_names":{"image_base64":None},
            },
            "role":"user"
        }
        if original_message is not None:
            temp_message["metadata"] = {**original_message["metadata"],**temp_message["metadata"]}
        return temp_message   
    
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
        
        message = messages[-1]                                    
            
        formated_prompt = PromptTemplate.from_template(VisualizationAgent.DEFAULT_USER_MESSAGE).format(                
            question=message["content"],
        ) 
        
        temp_message = modify_message_content(message,formated_prompt)  
        temp_message2 = modify_last_message(messages,temp_message)
        
        _,output = self.generate_llm_reply(raw_message,temp_message2,sender)            
        # ask the code agent to execute the code       
        self.send(message=self.create_temp_message(output,temp_message),recipient=self.code_agent)

        # summarize the conversation so far  
        last_message = self._messages[get_agent_name(self.code_agent)][-1]                
        if last_message["metadata"].get("code",0) != 0:
            return True, {"content":f'''FAIL TO GNERATE CODE''',"metadata":{"TERMINATE":True,"code":1}}
        # give the result to the user             
        response:ChatResponse = last_message["metadata"]["execute_result"]
        base64_image = response.variables["image_base64"]

        return True, {"content":base64_image,"metadata":{"TERMINATE":True}}
                 
        