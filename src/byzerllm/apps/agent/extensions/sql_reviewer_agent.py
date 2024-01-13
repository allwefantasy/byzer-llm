from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,message_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from .. import get_agent_name,run_agent_func,ChatResponse
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    
class SQLReviewerAgent(ConversableAgent): 
    DEFAULT_SYSTEM_MESSAGE='''你的任务是检查用户发送给你的Spark SQL语句。

请仔细阅读用户发给你 Spark SQL ，按下面的约束进行逐条检查：

1. SQL 中的所有字段和别名必须都使用 `` 括起来了                                          
2. SQL中不允许提到将一些变量替换成用户输入的内容，或者需要用户手动输入一些内容。
3. SQL中不允许出现诸如 ?, ?, ..., ? 这种参数化查询
4. SQL中不需有诸如 "有取 M 条记录， Top N类" 这种描述性的内容

注意：不要尝试生成任何代码去解决问题，你需要根据你的知识和经验，对用户发给你的代码进行审查。
                                          
如果违反了上述任何一个要求，你需要告诉用户，对应的SQL语句是具体不符合哪个要求。
如果符合上述所有要求，可以说：我觉得当前的代码没有什么问题了。
'''
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval, 
        chat_name:str,
        owner:str,               
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

        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], SQLReviewerAgent.generate_review_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 

    def generate_review_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
        ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  

        if messages is None:
            messages = self._messages[get_agent_name(sender)]

        message = messages[-1]                
        v = self.llm.chat_oai(conversations=message_utils.padding_messages_merge(self._system_message + messages[-1:]))
        new_message = {"content":v[0].output,"metadata":{}}        
        return True, message_utils.copy_error_count(message,new_message)

        