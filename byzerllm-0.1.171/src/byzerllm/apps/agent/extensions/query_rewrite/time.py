from typing import List,Dict,Annotated,Any
from byzerllm.utils.client import ByzerLLM,message_utils
from byzerllm.utils.retrieval import ByzerRetrieval
import copy
import pydantic
from datetime import datetime
from . import QueryRewriteResult,Action

class QueryTime:
    '''
    this tool is used to extract the key messages from the user's question which
    is used to generate the sql code conditions e.g. filter conditions, group by conditions, order by conditions
    or aggregate fields.
    '''
    def __init__(self,llm:ByzerLLM, retrieval:ByzerRetrieval,
                 sys_message:List[Dict[str,Any]],
                 messages:List[Dict[str,Any]],**kwargs):
        self.messages = messages
        self._system_message = sys_message
        self.llm = llm
        self.retrieval = retrieval
        self.params = kwargs

    def apply(self)->QueryRewriteResult:        
        m = copy.deepcopy(self.messages[-1])
        time_msg = ""  
            
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
            now = datetime.now().strftime("%Y-%m-%d")            
            class TimeRange(pydantic.BaseModel):
                '''
                时间区间
                格式需要如下： yyyy-MM-dd
                '''  
                
                start: str = pydantic.Field(...,description="开始时间.时间格式为 yyyy-MM-dd")
                end: str = pydantic.Field(...,description="截止时间.时间格式为 yyyy-MM-dd") 

            t = self.llm.chat_oai(conversations=[{
                "content":f'''当前时间是 {now}。根据用户的问题，计算时间区间。时间格式为 yyyy-MM-dd。用户的问题是：{t[0].value.time}''',
                "role":"user"
            }],response_class=TimeRange)
            
            if t[0].value and t[0].value.start and t[0].value.end:
                time_range:TimeRange = t[0].value
                time_msg = f'''时间区间是：{time_range.start} 至 {time_range.end}'''  
                print(f'compute the time range:{time_msg} {m["content"]}\n\n',flush=True)                                    

        return QueryRewriteResult(message = m,action=Action.CONTINUE,extra_info={"time_msg":time_msg})        
             


    