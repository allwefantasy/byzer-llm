from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,message_utils,code_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from .. import get_agent_name,run_agent_func,ChatResponse
import json
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    
class SQLReviewerAgent(ConversableAgent): 
    DEFAULT_SYSTEM_MESSAGE='''你的任务是检查用户发送给你的Spark SQL语句。
请仔细阅读用户发给你 Spark SQL，具体请按如下方式步骤进行检查，务必一步一步来：

1. 仔细检查SQL中的每一个字段和别名是否都使用反引号（backtick）或者反撇号（grave accent）`括起来。
2. SQL中不允许有文字描述提到让用户用户手动输入的内容，比如 "请输入"，"请填写" 等。
3. SQL中不允许出现诸如 ?, ?, ..., ? 这种参数化查询
4. SQL中不需有诸如 "有取 M 条记录， Top N类" 这种描述性的内容

请输出每一步检查结果，并说明原因。  
                                          
最后，对上面的检查结果和原因重新以json数组格式输出：

```json
[{
    id： 检查步骤序号,
    pass: 是否通过，true/false,
    reason: 原因                                    
}]                      
```
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
        print(f'''
review code:
    {message}. 
review result:
    {v[0].output}
''',flush=True)
        checks = json.loads(code_utils.extract_code(v[0].output)[-1][1])

        c = []
        for check in checks:
            if not check["pass"]:
                c.append(check["reason"])

        if len(c) > 0:
            t = "\n".join(c)
            new_message = {"content":f'''代码存在一些问题，具体的问题如下：\n{t}''',"metadata":{}}
            return True, message_utils.copy_error_count(message,new_message)
        
        t = self.llm.chat_oai(conversations=[
    {"role":"user",
    "content":f'''仔细检查下面的 Spark SQL：

{message["content"]}

请找出所有的字段以及别名，去掉函数，保留反引号，并将他们按照出现顺序，以json数组格式输出：

```json
[
  "字段或者别名"
]
```
'''}])
        try:
            fields = json.loads(code_utils.extract_code(t[0].output)[-1][1])
            for field in fields:
                for field in fields:            
                    if "`" not in field:
                        if f"`{field}`" not in message["content"]:
                            new_message = {"content":f'''代码存在问题，字段或者别名: {field} 没有被反引号括起来,请修改''',"metadata":{}}            
                            return True, message_utils.copy_error_count(message,new_message)
        except Exception:
            pass
        
        new_message = {"content":f'''代码没有问题，可以正常允许。''',"metadata":{}}                        
                     
        return True, message_utils.copy_error_count(message,new_message)

        