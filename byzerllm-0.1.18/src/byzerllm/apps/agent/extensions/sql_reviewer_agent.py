from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM
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
    DEFAULT_SYSTEM_MESSAGE='''You are a helpful AI assistant. You are also a Spark SQL expert. 你的任务是检查一段包含了Spark SQL 的文本，该文本
会使用 ```sql ``` 语法来标记 Spark SQL 代码。你需要检查这段文本中的 Spark SQL 代码是否符合以下要求：    

1. 这段话里不允许提到将一些变量替换成用户输入的内容，或者需要用户手动输入一些内容。
2. 不允许在SQL 中出现诸如 ?, ?, ..., ? 类似这种参数化查询
3. SQL 代码必须使用 ```sql ``` 语法来标记，不能出现 ```vbnet ``` 等语法标记
4. 确保这些 SQL 无需任何修改即可运行，所以不能有取 M 条记录， Top N类，需要去掉这些约束。

如果违反了上述任何一个要求，请给出建议内容，但不要尝试生成任何SQL代码，仅仅给出建议。

在以下情况，你需要直接结束对话：

1. 如果用户给的Spark SQL 代码已经符合了所有要求。
2. 如果你觉得看起来没有什么问题了。
3. 如果用户给的内容不包含任何SQL代码。
4. 如果用户给的内容是诸如"很高兴能帮到您！如果您还有其他问题或需要进一步的帮助，请随时告诉我。祝您工作顺利！"。

结束对话方式为： 直接输出 TERMINATE 字符串。不要有任何其他字符。

注意：
1. 不要翻译 TERMINATE 为中文。
2. 仅仅给出建议，不要尝试生成任何SQL代码。
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

        