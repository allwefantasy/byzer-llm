from typing import List,Dict,Annotated,Any
from byzerllm.utils.client import ByzerLLM,message_utils
from byzerllm.utils.retrieval import ByzerRetrieval
import copy
from . import QueryRewriteResult,Action

class QueryRhetorical:
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
        ## extract key messages is the user want to generate sql code
        def reply_with_clarify(content:Annotated[str,"不理解问题，反问用户的内容"]): 
            '''
            对问题如果不清晰(不包括时间问题)，无法抽取出有效的关键信息，那么可以调用该函数，反问用户。
            '''
            return content 

        def reply_with_key_messages(content:Annotated[list[str],"列表形式的关键信息,诸如过滤条件，指标，分组条件"]):  
            '''
            如果你能抽取出有效的关键信息，那么可以调用该函数
            '''      
            return content 

        
            
        temp_conversation = [{"role":"user","content":f'''
首先根据我的问题，关联前面的对话，针对当前的问题以列表形式罗列我问题中的关键信息,诸如过滤条件，指标，分组条件。不需要生成SQL。
注意:
* 不要考虑时间
* 如果补充信息和原始问题有冲突，以原始信息为准
* 务必要参考对话中我们提及的表结构信息，示例数据，枚举值等，以便对表进行正确的过滤，分组，排序等操作。
* 过滤条件中的字段的值如果有不符合枚举值的，可以自动修正为枚举值里的值，如果无法修正，则不要添加该过滤条件。
* 生成的SQL中的字段务必要出现在前面对话中提及的表结构信息中的schema里。 
        '''}] 

         

        t = self.llm.chat_oai(conversations=message_utils.padding_messages_merge(self._system_message  + self.messages + self.params.get("temp_conversation",temp_conversation)),
                                tools=[self.params.get("reply_with_clarify",reply_with_clarify) ,
                                       self.params.get("reply_with_key_messages",reply_with_key_messages) ],
                                execute_tool=True)
        action = Action.CONTINUE
        if t[0].values:             
            v = t[0].values[0]
            if isinstance(v,str):
                print("invoke reply_with_clarify",flush=True)
                m = {"content":v,"metadata":{"TERMINATE":True}}
                action = Action.STOP
            
            if isinstance(v,list):
                print("invoke reply_with_key_messages",flush=True)
                v = " ".join(v)          
                key_msg = v
                print(f'compute the key info:{m["content"]}\n\n',flush=True)
                old_content = m["content"]
                m["content"] = f'''补充信息：{time_msg} {key_msg} \n原始问题：{old_content} '''
                print(f'final query:{m["content"]}\n\n',flush=True)                                    

        return QueryRewriteResult(message = m,action = action,extra_info={"key_msg":key_msg})        
             


    