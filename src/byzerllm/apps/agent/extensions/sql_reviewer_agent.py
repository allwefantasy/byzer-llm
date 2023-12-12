from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,ByzerRetrieval
from ..agent import Agent
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
from .. import get_agent_name,run_agent_func,ChatResponse
from ....utils import generate_str_md5
from byzerllm.utils.client import TableSettings,SearchQuery,LLMHistoryItem,LLMRequest
import uuid
import json
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter,Document
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    
class SQLReviewerAgent(ConversableAgent): 
    DEFAULT_SYSTEM_MESSAGE='''You are a helpful AI assistant. You are also a Spark SQL expert. 你的任务是检查一段包含了Spark SQL 的文本，该文本
会使用 ```sql ``` 语法来标记 Spark SQL 代码。你需要检查这段文本中的 Spark SQL 代码是否符合以下要求：    

1. 是否有需要手动输入或者修改的地方？如果有，请指出需要修改的地方，并且给出修改的建议。比如，一段 Spark SQL 里有 in 查询，查询里用了一些占位符，需要人工修改，那么你应该建议使用子查询来替代。

如果符合上述要求，那么请直接在最后回复 TERMINATE 以结束对话。
'''
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
                        
        _,v = self.generate_llm_reply(raw_message,messages,sender)
        
        return True, {"content":v,"metadata":{}}

        