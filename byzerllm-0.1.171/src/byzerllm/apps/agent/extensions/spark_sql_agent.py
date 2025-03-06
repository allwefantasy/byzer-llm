from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union,Annotated
from ....utils.client import ByzerLLM,code_utils,message_utils,parallel_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
import numpy as np
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import re
from .. import get_agent_name,run_agent_func,ChatResponse
from ....utils import generate_str_md5
from byzerllm.utils.client import LLMHistoryItem,LLMRequest
import json
from byzerllm.apps.agent.extensions.simple_retrieval_client import SimpleRetrievalClient
import pydantic
from datetime import datetime
from byzerllm.apps.agent.extensions.query_rewrite.context import QueryContext
from byzerllm.apps.agent.extensions.query_rewrite.condition import QueryCondition
from byzerllm.apps.agent.extensions.query_rewrite.time import QueryTime
from byzerllm.apps.agent.extensions.query_rewrite import Action
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    

class SparkSQLAgent(ConversableAgent): 
    DEFAULT_SYSTEM_MESSAGE='''你非常精通 Spark SQL, 并且能够根据用户的问题，基于提供的表信息，生成对应的 Spark SQL 语句。

下面是你具备的一些能力：

### 联系上下文分析
                                                                                                                               
当面对用户的问题时，要多回顾过往的对话，根据上下文获取补充信息，去理解用户的需求。

示例:
1. 用户问题： 2023别克君威的销量是多少？
2. 回答： 2023年别克君威的销量是 1000 辆
3. 用户问题： 2024年呢？

此时，无需再询问用户查询什么，结合上一次提问的内容，来理解问题。
结合上一次用户提的问题，用户的实际问题是： 2024别克君威的销量是多少？
这个时候再进一步生成SQL语句。

学习我上面的示例，拓展到其他的场景。

### 时刻结合用户给出的表信息来修正查询

通常用户会给出表的信息包括：
1. 表的名字和结构schema
2. 表的一些统计信息，比如表的字段枚举值等
3. 表的示例数据

### 日期处理能力

当你生成 SQL 时，涉及到日期字段，你需要参考表的 Schema 信息，自动将用户的日期表达式转换成表的日期格式。如果表中
使用多个字段来提供日期信息，比如年，月，日，优先使用他们，而不是使用复杂的日期格式。

### 其他能力

诸如 会根据用户的问题，自动分析出用户的查询意图，然后生成对应的SQL语句。                                                                                                                                                                         

特别需要注意的是：
1. 你生成的代码要用 SQL 代码块包裹，```sql\n你的代码```, 注意一定要Block需要用sql标注而非vbnet。
3. 生成的 Spark SQL 语句中，所有字段或者别名务必需要用反引号 `` 括起来，尤其是 as 关键字后面的别名。
4. 任何情况下都不要拆分成多段代码输出，请一次性生成完整的代码片段，确保代码的完整性。
'''
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval, 
        chat_name:str,
        owner:str,                            
        sql_reviewer_agent: Union[Agent, ClientActorHandle,str],
        byzer_engine_agent: Union[Agent, ClientActorHandle,str],      
        retrieval_cluster:str="data_analysis",
        retrieval_db:str="data_analysis",   
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,        
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[Dict, bool]] = False,
        byzer_url="http:://127.0.0.1:9003/run/script",
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
        
        self.retrieval_cluster = retrieval_cluster
        self.retrieval_db = retrieval_db
        self.chat_name = chat_name
        self.owner = owner
        self.simple_retrieval_client = SimpleRetrievalClient(llm=self.llm,
                                                        retrieval=self.retrieval,
                                                        retrieval_cluster=self.retrieval_cluster,
                                                        retrieval_db=self.retrieval_db,
                                                        ) 
        self.byzer_url = byzer_url        
        self.sql_reviewer_agent = sql_reviewer_agent
        self.byzer_engine_agent = byzer_engine_agent
        self._reply_func_list = []                
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], SparkSQLAgent.generate_sql_reply) 
        self.register_reply([Agent, ClientActorHandle,str], SparkSQLAgent.generate_execute_sql_reply)
        self.register_reply([Agent, ClientActorHandle,str], SparkSQLAgent.generate_reply_for_reviview)        
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
         
        m = messages[-1]        

        if not m.get("metadata",{}).get("skip_long_memory",False): 
            # recall old memory and update the system prompt
            old_memory = self.simple_retrieval_client.search_content(q=m["content"],owner=self.owner,url="rhetorical",limit=3)
            if len(old_memory) != 0:
                c = json.dumps(old_memory,ensure_ascii=False)
                self.update_system_message(f'''{self.system_message}\n

下面是用户的一些行为偏好，在回答问题的时候，可以参考：
```json
{c}                                       
```  
''') 

        # query rewrite                        
        if len(re.sub(r'\s+', '', m["content"])) < 60:

            # context query rewrite
            rewriter = QueryContext(self.llm,self.retrieval,self._system_message,messages)
            r = rewriter.apply()
            m["content"] = r.extra_info["new_query"]

            # time query rewrite
            rewriter = QueryTime(self.llm,self.retrieval,self._system_message,messages)
            r = rewriter.apply()
            time_msg = r.extra_info["time_msg"]
            
            # structure query rewrite            
            rewriter = QueryCondition(self.llm,self.retrieval,self._system_message,messages,
                                      time_msg=time_msg)
            r = rewriter.apply()

            # # we may need more information from the user
            # if r.action == Action.STOP:
            #     return True, r.message
            
            key_msg = r.extra_info["key_msg"] 
            m["content"] = f'''补充信息：{time_msg} {key_msg} \n原始问题：{m["content"]} '''                       
        
        
        # try to awnser the user's question or generate sql
        
        temp_conversation = []
        _,v = self.generate_llm_reply(raw_message,message_utils.padding_messages_merge(messages+temp_conversation),sender)
        codes = code_utils.extract_code(v)
        has_sql_code = code_utils.check_target_codes_exists(codes,["sql"])         

        # if we have sql code, ask the sql reviewer to review the code         
        if has_sql_code: 
            
            # sync the question to the sql reviewer                           
            self.send(message_utils.un_termindate_message(messages[-1]),self.sql_reviewer_agent,request_reply=False)
            
            # send the sql code to the sql reviewer to review            
            self.send({
                    "content":f'''
```sql
{code_utils.get_target_codes(codes,["sql"])[0]}
```
'''},self.sql_reviewer_agent)
            
            # get the sql reviewed.             
            conversation = message_utils.un_termindate_message(self.chat_messages[get_agent_name(self.sql_reviewer_agent)][-1])            
            
            if conversation["content"] == "FAIL TO GENERATE SQL CODE":
                return True, {"content":f'Fail to generate sql code.',"metadata":{"TERMINATE":True}}
            
            # send the sql code to the byzer engine to execute
            print(f"send the sql code to the byzer engine to execute {conversation}",flush=True)
            self.send(message=conversation,recipient=self.byzer_engine_agent)  
            
            execute_result = self.chat_messages[get_agent_name(self.byzer_engine_agent)][-1]             
            print(f"execute_result: {execute_result}",flush=True)
            
            if message_utils.is_success(execute_result):
                return True,{"content":execute_result["content"],"metadata":{"TERMINATE":True,"rewrite_query":m["content"],"sql":conversation}}
            else:
                return True,{"content":f'Fail to execute the analysis. {execute_result["content"]}',"metadata":{"TERMINATE":True}}

        return True,  {"content":v,"metadata":{"TERMINATE":True}}
    
    def generate_execute_sql_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
        ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]: 
        
        if get_agent_name(sender) != get_agent_name(self.byzer_engine_agent):
            return False, None
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)] 
                
        message = messages[-1]
        if message["metadata"]["code"] == 0:
            return True, None
                
        if message_utils.check_error_count(message,max_error_count=3):
            return True, None
        
        last_conversation = [{"role":"user","content":'''请根据上面的错误，修正你的代码。注意，除了修正指定的错误以外，请确保 SQL 语句其他部分不要变更。'''}]   
        t = self.llm.chat_oai(conversations=message_utils.padding_messages_merge(self._system_message + messages + last_conversation))
        _,new_code = code_utils.extract_code(t[0].output)[0]
        new_message = {"content":f'''
```sql
{new_code}
```
'''}    
        message_utils.copy_error_count(message,new_message)
        return True, message_utils.inc_error_count(new_message)

        
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

        target_message = {
            "content":"FAIL TO GENERATE SQL CODE",
            "metadata":{"TERMINATE":True},
        }    
        
        # check if code is passed or not by the sql reviewer
        
        def run_code():  
            '''
            用户表达肯定的观点或者代码没有问题，请调用我
            '''              
            return 0        
    
        def ignore():  
            '''
            用户表达否定或者代码不符合特定规范，可以调用我
            '''              
            return 1    
        

        last_conversation = messages[-1] 
        temp_conversation = {
            "role":"user",
            "content":"注意，你只需要判断调用哪个函数，并不需要解决问题。",
        }    

        ts= parallel_utils.chat_oai(self.llm,1,
                                conversations=message_utils.padding_messages_merge([last_conversation,temp_conversation]),
                                tools=[run_code,ignore],
                                execute_tool=True)
        t = None
        for temp in  ts:
            if temp[0].values:
                t = temp
                break

        # t = self.llm.chat_oai(conversations=[last_conversation],
        #                   tools=[run_code,ignore],
        #                   execute_tool=True)  
        
        if t and t[0].values:               
            if t[0].values[0] == 0:
                target_message["content"] = messages[-2]["content"]                
            else:   
                print(f"Fail to pass the review: {last_conversation}. Try to regenerate the sql",flush=True)             
                t = self.llm.chat_oai(conversations=message_utils.padding_messages_merge(self._system_message + messages+[{
                    "content":'''请修正你的代码。注意，除了修正指定的错误以外，请确保 SQL 语句其他部分不要变更,代码需要用 ```sql```包裹起来。''',
                    "role":"user"
                }]))
                print(f"Try to regenerate the sql: {t[0].output}",flush=True)
                sql_codes = code_utils.get_target_codes(code_utils.extract_code(t[0].output),["sql"])
                if sql_codes:
                    target_message["content"] = sql_codes[0]
                    target_message["metadata"]["TERMINATE"] = False
                    message_utils.inc_error_count(target_message)
        else:        
            print(f"Fail to recognize the reveiw result: {last_conversation}",flush=True)
        ## make sure the last message is the reviewed sql code    
        return True, target_message   

        
    
        
        



            

        