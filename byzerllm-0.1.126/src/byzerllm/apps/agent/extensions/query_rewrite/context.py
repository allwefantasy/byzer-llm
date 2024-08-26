from typing import List,Dict,Annotated,Any
from byzerllm.utils.client import ByzerLLM,message_utils,code_utils,LLMRequest
from byzerllm.utils.retrieval import ByzerRetrieval
import json
import numpy as np
import copy
import pydantic
from . import QueryRewriteResult,Action

class QueryContext:
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

    def apply(self):        
        m = copy.deepcopy(self.messages[-1])
        temp_conversation = [{"role":"user","content":'''
首先，根据我们前面几条聊天内容，针对我现在的问题，进行一个信息扩充。如果我的问题信息已经比较充足，
则输出原有问题即可。
     
注意：
1. 不要询问用户问题          
2. 不要生成SQL
3. 不要额外添加上下文中不存在的信息
4. 不要关注时间,不要改写时间
6. 尽量保证信息完整 
'''}]
        class SingleLine(pydantic.BaseModel):
            content:str=pydantic.Field(...,description="改写后的query")

        t = self.llm.chat_oai(
            conversations=message_utils.padding_messages_merge(self._system_message + self.messages + self.params.get("temp_conversation",temp_conversation)),
            response_class=SingleLine,
            enable_default_sys_message=True
            ) 
        new_query = m["content"]           
        if t[0].value:
            new_query = t[0].value.content               
            print(f'context query rewrite: {m["content"]} -> {new_query}\n\n',flush=True)          
            m["content"] = new_query   
    #         if new_query != m["content"]:
    #             temp1 = self.llm.emb(None,LLMRequest(instruction=new_query))
    #             temp2 = self.llm.emb(None,LLMRequest(instruction=m["content"]))
    #             sim = np.dot(temp1[0].output,temp2[0].output)
    #             if sim > 0.8:
    #                 print(f'context query rewrite: {m["content"]} -> {new_query}\n\n',flush=True)
    #                 m["content"] = new_query                    
    #             else:
    #                 print(f'''context query rewrite fail. 
    # the similarity is too low {sim}
    # query:  {m["content"]}
    # new_query: {new_query}
    # \n\n''',flush=True)

        return QueryRewriteResult(message = m,action = Action.CONTINUE,extra_info={"new_query":new_query})       
             


    