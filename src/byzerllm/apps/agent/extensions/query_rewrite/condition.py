from typing import List,Dict,Annotated,Any
from byzerllm.utils.client import ByzerLLM,message_utils,code_utils
from byzerllm.utils.retrieval import ByzerRetrieval
import copy
import json
from . import QueryRewriteResult,Action

class QueryCondition:
    '''
    this tool is used to extract the key messages from the user's question which
    is used to generate the sql code conditions e.g. filter conditions, group by conditions, order by conditions
    or aggregate fields.
    '''
    def __init__(self,llm:ByzerLLM, retrieval:ByzerRetrieval,
                 sys_message:List[Dict[str,Any]],
                 messages:List[Dict[str,Any]],
                 **kwargs):
        self.messages = messages
        self._system_message = sys_message
        self.llm = llm
        self.retrieval = retrieval
        self.params = kwargs        

    def apply(self)->QueryRewriteResult:        
        m = copy.deepcopy(self.messages[-1])
        time_msg = self.params.get("time_msg","")  
        key_msg = ""      
        temp_conversation = [{"role":"user","content":'''
首先根据我的问题，关联前面的对话，尤其是前面的表结构表结构信息，示例数据，表统计信息等，找到我当前问题中的关键信息,
诸如过滤条件，指标，分组条件。不需要生成SQL。                      

具体请按如下方式步骤补充信息，务必一步一步来：

1. 回顾前面的会话，对提到的表结构信息，示例数据，表统计信息等进行回顾，列出字段列表。
2. 对当前问题进行拆解，找到可能的过滤条件，指标，分组条件等。以 字段名称=值 的形式罗列出来。
3. 根据表统计信息中列表，对第二步过滤条件的值进行修正。具体做法是，如果过滤条件字段在表统计信息中有枚举值，
   检查过滤字段的值是否在枚举值中，如果不在，找到最接近的枚举值，修正过滤条件的值。
4. 对最后修正的结果，重新以 Json 格式进行输出

```json
[{
    "表字段名称"："从问题得到的值"
}]                      
```                                            
请输出每一步的结果。                     
'''}] 

         

        t = self.llm.chat_oai(conversations=message_utils.padding_messages_merge(
            self._system_message  + self.messages + self.params.get("temp_conversation",temp_conversation)))
        action = Action.CONTINUE
        try:
            info = json.loads(code_utils.extract_code(t[0].output)[-1][1])
            for item in info:
                for k,v in item.items():
                    key_msg += f''' {k}={v}'''
        except Exception:
            print(f"QueryCondition error: {t[0].output}",flush=True)
            pass

        old_content = m["content"]
        m["content"] = f'''补充信息：{time_msg} {key_msg} \n原始问题：{old_content} '''
        print(f'final query:{m["content"]}\n\n',flush=True)                                              
        return QueryRewriteResult(message = m,action = action,extra_info={"key_msg":key_msg})        
             


    