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
首先根据我的问题，关联前面的对话，尤其是前面的表结构表结构信息，示例数据，枚举值等，找到我当前问题中的关键信息,
诸如过滤条件，指标，分组条件。不需要生成SQL。                      

具体请按如下方式步骤补充信息，务必一步一步来：

1. 回顾前面的会话，对提到的表结构信息，示例数据，枚举值等进行回顾，找到可能和当前这个问题相关的信息。
2. 对当前问题进行拆解，找到可能的过滤条件，指标，分组条件等。以 字段名称=值 的形式罗列出来。
3. 根据枚举值，示例数据等信息，对值进行修正，比如如果某个值是枚举值里的缩写，那么将该值修改为完整的值，其他值也参考类似方法来完成修正。
4. 对最后修正的结果，重新以 Json 格式进行输出

```json
[{
    "表字段名称"："从问题得到的值"
}]                      
```                                            
请输出每一步的结果。                      
'''}] 

         

        t = self.llm.chat_oai(conversations=message_utils.padding_messages_merge(self._system_message  + self.messages + self.params.get("temp_conversation",temp_conversation)))
        action = Action.CONTINUE
        try:
            info = json.loads(code_utils.extract_code(t[0].output)[-1][1])
            for k,v in info.items():
                key_msg += f''' {k}={v}'''
        except Exception:
            pass

        old_content = m["content"]
        m["content"] = f'''补充信息：{time_msg} {key_msg} \n原始问题：{old_content} '''
        print(f'final query:{m["content"]}\n\n',flush=True)                                              
        return QueryRewriteResult(message = m,action = action,extra_info={"key_msg":key_msg})        
             


    