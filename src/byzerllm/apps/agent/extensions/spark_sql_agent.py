from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,code_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
from .. import get_agent_name,run_agent_func,ChatResponse
from ....utils import generate_str_md5
from byzerllm.utils.client import LLMHistoryItem,LLMRequest
import uuid
import json
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter,Document
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    
class SparkSQLAgent(ConversableAgent): 
    DEFAULT_SYSTEM_MESSAGE='''You are a helpful AI assistant. You are also a Spark SQL expert. 

In the following cases, suggest Spark SQL code (in a sql coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put -- filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.

既然你是一个 Spark SQL 专家，并且你总是对问题进行拆解，一步一步使用 Spark SQL 完成任务，那么为了能够让SQL之间能够进行衔接，你需要使用 with 语句。

在 Spark SQL 中，WITH 子句通常用于定义临时视图（Temporary Views），这些视图在当前 SQL 查询中是可见的。以下是一个使用 WITH 子句的例子：

假设我们有一个名为 employees 的表，其中包含 id, name, 和 department_id 字段。我们想要计算每个部门的员工数量。

SQL 查询如下：

```sql
WITH department_count AS (
  SELECT department_id, COUNT(*) as employee_count
  FROM employees
  GROUP BY department_id
)
SELECT d.department_id, d.employee_count
FROM department_count d;
```

在这个例子中，WITH 子句创建了一个名为 department_count 的临时视图，该视图包含每个部门的 department_id 和相应的 employee_count（员工数量）。随后的 SELECT 语句从这个临时视图中检索数据。

请注意： 生成的 Spark SQL 要避免了手动输入，而是让数据库自动为我们处理，比如 in 查询里，要用子查询，而不是手动输入。
The last but most important, let me know if you have any areas of confusion. If you do, please don't generate code, ask me, and provide possible solutions.



    '''
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval,        
        code_agent: Union[Agent, ClientActorHandle,str],  
        sql_reviewer_agent: Union[Agent, ClientActorHandle,str],      
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
        self.sql_reviewer_agent = sql_reviewer_agent
        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], SparkSQLAgent.generate_reply_for_reviview)
        self.register_reply([Agent, ClientActorHandle,str], SparkSQLAgent.generate_sql_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 

    def generate_sql_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
        ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  

        if get_agent_name(sender) == get_agent_name(self.sql_reviewer_agent):
            return False,None
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)]

        # give the response to    
        _,v = self.generate_llm_reply(raw_message,messages,sender)
        codes = code_utils.extract_code(v)
        has_sql_code = False

        for code in codes:                  
            if code[0]!="unknown":                
                has_sql_code = True           

        if has_sql_code:                
            self.send(messages[-1],self.sql_reviewer_agent,request_reply=False)
            self.send({
                    "content":v
                },self.sql_reviewer_agent)
            
            reply = self.chat_messages[get_agent_name(self.sql_reviewer_agent)][-2]["content"]
            # reply = run_agent_func(self.sql_reviewer_agent,"get_chat_messages")[get_agent_name(self.sql_reviewer_agent)][-2]        
            return True, {"content":reply,"metadata":{"TERMINATE":True}}             
        
        return True,  {"content":v,"metadata":{"TERMINATE":True}}
        
    def generate_reply_for_reviview(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
        ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]: 
        
        if get_agent_name(sender) != get_agent_name(self.sql_reviewer_agent):
            return False, None
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)] 

        _,v = self.generate_llm_reply(raw_message,messages,sender)
        
        return True, {"content":v,"metadata":{}}
        
        



            

        