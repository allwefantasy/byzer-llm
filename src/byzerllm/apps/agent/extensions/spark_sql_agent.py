from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union,Annotated
from ....utils.client import ByzerLLM,code_utils,message_utils
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
from byzerllm.apps.agent.extensions.simple_retrieval_client import SimpleRetrievalClient
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter,Document
import pydantic

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    

class SparkSQLAgent(ConversableAgent): 
    DEFAULT_SYSTEM_MESSAGE='''You are a helpful AI assistant. You are also a Spark SQL expert. 
你总是要多联系上下文对问题做分析，同时根据上下文进行问题的拆解，先给出详细解决问题的思路，最后确保你生成的代码都在一个 SQL Block 里。类似下面格式：

```sql
你生成的SQL代码
```

下面是联系上下文分析的一个示例:
1. 当我第一次问： 2023别克君威的销量是多少？
2. 你通过生成SQL然后回答了我的问题。
3. 当我再次问： 2024年呢？
4. 这里，你需要结合我上一次提问的内容，来理解问题，你可以理解为： 2024别克君威的销量是多少？

特别需要注意的是：
1. 你生成的Block需要用sql标注而非vbnet
2. 生成的 Spark SQL 语句中，所有字段或者别名务必需要用 `` 括起来。
'''
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval, 
        chat_name:str,
        owner:str,                            
        sql_reviewer_agent: Union[Agent, ClientActorHandle,str],      
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
        self._reply_func_list = []                
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], SparkSQLAgent.generate_sql_reply) 
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

        # recall old memory and update the system prompt
        old_memory = self.simple_retrieval_client.search_content(q=m["content"],owner=self.owner,url="rhetorical",limit=3)
        if len(old_memory) != 0:
            c = json.dumps(old_memory,ensure_ascii=False)
            self.update_system_message(f'''{self.DEFAULT_SYSTEM_MESSAGE}\n下面是我们以前对话的内容总结:
```json
{c}                                       
```  
你在回答我的问题的时候，可以参考这些内容。''') 

        time_msg = ""
            
        # to compute the real time range, notice that 
        # we will chagne the message content  

        class Item(pydantic.BaseModel):
            '''
            查询参数    
            如果对应的参数不符合字段要求，那么设置为空即可
            '''
            other: str = pydantic.Field(...,description="其他参数，比如用户的名字，或者其他的一些信息")
            time:  str = pydantic.Field(...,description="时间信息,比如内容里会提到天， 月份，年等相关词汇")


        t = self.llm.chat_oai([{
            "content":f'''{m["content"]}''',
            "role":"user"    
        }],response_class=Item)                 

        if t[0].value and t[0].value.time:     
            
            def calculate_time_range():
                '''
                计算时间的区间，注意这个函数没有参数。 请使用dateutil库。
                如果用户提到的是单一日期，那么都是按天为单位来进行区间计算。
                比如说：上个月，那么就是上个月的第一天到最后一天。
                去年11月份，那么就是去年11月份的第一天到最后一天。
                如果用户提到的是一个时间区间，那么就是按照用户提到的时间区间来进行计算。
                比如 去年三月到五月，那么就是去年三月的第一天到去年五月的最后一天。
                '''
                pass 

            class TimeRange(pydantic.BaseModel):
                '''
                时间区间
                格式需要如下： yyyy-MM-dd
                '''  
                
                start: str = pydantic.Field(...,description="开始时间.时间格式为 yyyy-MM-dd")
                end: str = pydantic.Field(...,description="截止时间.时间格式为 yyyy-MM-dd")                 
            
            t = self.llm.chat_oai([{
                "content":t[0].value.time,
                "role":"user"    
            }],impl_func=calculate_time_range,response_class=TimeRange,execute_impl_func=True)

            if t[0].value:
                time_range:TimeRange = t[0].value
                time_msg = f'''时间区间是：{time_range.start} 至 {time_range.end}'''  
                print(f'compute the time range:{m["content"]}\n\n',flush=True) 
        
        old_content = m["content"]
        if time_msg:
            m["content"] = f'''补充信息：{time_msg} \n原始问题：{old_content} '''

        key_msg = ""
        ## extract key messages is the user want to generate sql code
        def reply_with_clarify(content:Annotated[str,"不理解问题，反问用户的内容"]): 
            '''
            对问题如果不清晰，无法抽取出有效的关键信息，那么可以调用该函数，反问用户。
            '''
            return content 

        def reply_with_key_messages(content:Annotated[list[str],"列表形式的关键信息,诸如过滤条件，指标"]):  
            '''
            如果你能抽取出有效的关键信息，那么可以调用该函数
            '''      
            return content 

            
        last_conversation = [{"role":"user","content":f'''
        首先根据我的问题，关联前面的对话，针对当前的问题以列表形式罗列我问题中的关键信息,诸如过滤条件，指标。不需要生成SQL。'''}]        
        t = self.llm.chat_oai(conversations=message_utils.padding_messages_merge(self._system_message  + messages + last_conversation),
                            tools=[reply_with_clarify,reply_with_key_messages],
                            execute_tool=True)

        if t[0].values:     
            v = t[0].values[0]
            if isinstance(v,str):
                return True,{"content":v,"metadata":{"TERMINATE":True}}
            
            if isinstance(v,list):
                v = " ".join(v)          
            key_msg = v
            print(f'compute the key info:{m["content"]}\n\n',flush=True)
        
        if key_msg:
            m["content"] = f'''补充信息：{key_msg} \n原始问题：{old_content} '''
        
        # check if the user's question is ambiguous or not, if it is, try to ask the user to clarify the question.                        
        # def reply_with_clarify(content:Annotated[str,"这个是你反问用户的内容"]):
        #     '''
        #     如果你不理解用户的问题，那么你可以调用这个函数，来反问用户。
        #     '''
        #     return content             
    
        # last_conversation = [{"role":"user","content":"\n请对我上面的问题进行思考，尝试理解。只有确实有歧义或者不明确的地方，才去调用上面的函数。"}]        
        # t = self.llm.chat_oai(conversations=message_utils.padding_messages_merge(self._system_message + messages + last_conversation),
        #                   tools=[reply_with_clarify],
        #                   execute_tool=True)
        
        # if t[0].values:               
        #     return True,{"content":t[0].values[0],"metadata":{"TERMINATE":True}}
                         
        
        # try to awnser the user's question or generate sql
        _,v = self.generate_llm_reply(raw_message,messages,sender)
        codes = code_utils.extract_code(v)
        has_sql_code = code_utils.check_target_codes_exists(codes,["sql"])         

        # if we have sql code, ask the sql reviewer to review the code         
        if has_sql_code: 
            # sync the question to the sql reviewer               
            
            self.send(messages[-1],self.sql_reviewer_agent,request_reply=False)
            # send the sql code to the sql reviewer to review
            
            self.send({
                    "content":code_utils.get_target_codes(codes,["sql"])[0],
                },self.sql_reviewer_agent)
            
            # get the sql reviewed.             
            conversation = self.chat_messages[get_agent_name(self.sql_reviewer_agent)][-1]   
            try:                     
                reply = self.execute_spark_sql(conversation["content"])
            except Exception as e:
                # get full exception
                import traceback
                reply = f"执行代码出错：{traceback.format_exc()} {e}"                
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

        target_message = {
            "content":"",
            "metadata":{"TERMINATE":True},
        }    

        def reply_with_review_success(sql:Annotated[str,"我们刚刚发送给用户的sql代码"]):
            '''
            如果用户觉得代码没问题，那么可以调用该函数
            '''            
            target_message["content"] = messages[-2]["content"]

        def reply_with_review_fail(content:Annotated[str,"根据用户反馈的问题，我们修正后的SQL代码"]):
            '''
            如果用户觉得代码还有问题时，调用该函数
            '''
            target_message["content"] = content
            target_message["metadata"]["TERMINATE"] = False    

        last_conversation = [{"role":"user","content":"请根据我的描述，决定是否调整你的代码。"}]
        
        print(message_utils.padding_messages_merge(self._system_message + messages + last_conversation))
        self.llm.chat_oai(conversations=message_utils.padding_messages_merge(self._system_message + messages + last_conversation),
                          tools=[reply_with_review_success,reply_with_review_fail],
                          execute_tool=True)  
        
        print(target_message,flush=True)              

        ## make sure the last message is the reviewed sql code    
        return True, target_message
    
    def execute_spark_sql(self,sql:Annotated[str,"Spark SQL 语句"])->str:
        '''
        执行 Spark SQL 语句
        '''
        
        print(f"execute spark sql: {sql}",flush=True)

        v = self.llm._rest_byzer_script(f"""
load csv.`file:///home/byzerllm/projects/jupyter-workspace/nlp2query/h.csv` where header="true" as test_table;
!profiler sql '''
{sql}                                        
''';
""",owner="william",url="http://192.168.1.248:9003/run/script")
        return json.dumps(v,ensure_ascii=False)

        
    
        
        



            

        