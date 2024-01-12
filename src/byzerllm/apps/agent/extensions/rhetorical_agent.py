from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,message_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from .. import get_agent_name,run_agent_func,ChatResponse
from byzerllm.apps.agent.extensions.simple_retrieval_client import SimpleRetrievalClient
import json
   
class RhetoricalAgent(ConversableAgent): 
    
    DEFAULT_SYSTEM_MESSAGE = '''你是一个非常善于反思和总结的AI助手。
尝试从之前的对话中，找到下面三种内容。

1. 用户指定你需要记住的一些内容。
2. 用户额外补充的一些信息。
3. 用户和你说明的一些名词定义。

这些信息是为了方便以后回答我的问题的时候，可以参考这些信息，避免我重复和你确认这些信息。

下面是一些例子，方便你理解。

假设我们有如下的对话：

```
用户： 上分的销售额是多少？
助手： 你是指上海分公司的销售额吗？
用户： 是上海分行的销售额。
```

你的总结：

```
上分是指上海分行
```

下次当我再谈到上分的时候，你看到上面的总结后，可以直接明确我指就是上海分行。

再比如，假设我们有如下的对话：

```
用户： 奔驰上个月的销量
助手： 销量是指销售额还是销售量？
用户： 销售额
```

你的总结：

```
询问 奔驰上个月的销量时，销量是指销售额。
如果在之前总结里，已经有提到的，就无需再次总结。
```
'''
    
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval,     
        chat_name:str,
        owner:str,           
        retrieval_cluster:str="data_analysis",
        retrieval_db:str="data_analysis",
        update_context_retry: int = 3,
        chunk_size_in_context: int = 1,
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
                        
        self.chat_name = chat_name
        self.owner = owner
        
        self.update_context_retry = update_context_retry
        self.chunk_size_in_context = chunk_size_in_context

        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], RhetoricalAgent.my_reply) 
        self.register_reply([Agent, ClientActorHandle,str], RhetoricalAgent.check_termination_and_human_reply) 
                
        self.retrieval_cluster = retrieval_cluster
        self.retrieval_db = retrieval_db         

        self.simple_retrieval_client = SimpleRetrievalClient(llm=self.llm,
                                                             retrieval=self.retrieval,
                                                             retrieval_cluster=self.retrieval_cluster,
                                                             retrieval_db=self.retrieval_db,
                                                             )         
                
          
    def my_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:          

        if messages is None:
            messages = self._messages[get_agent_name(sender)]  

        m = messages[-1]        
        
        old_conversations = self.simple_retrieval_client.search_content(q=m["content"],owner=self.owner,url="rhetorical",limit=3)
        if len(old_conversations) != 0:
            c = json.dumps(old_conversations,ensure_ascii=False)
            self.update_system_message(f'''{self.DEFAULT_SYSTEM_MESSAGE}\n下面是我们以前对话的内容总结:
```json
{c}                                       
```  
你在回答我的问题的时候，可以参考这些内容。''')
                         
        last_conversation = [{"role":"user","content":'''现在，开始回顾我们前面的对话，并且找到我指定你需要记住的一些内容，
或者我额外补充的一些信息，或者我和你说明的一些名词定义。'''}]

        c_messages = messages[-7:-1]
        # always choose the last six messages to generate the reply        
        _,v2 = self.generate_llm_reply(raw_message,message_utils.padding_messages_merge(self._system_message + c_messages[-7:-1] + last_conversation),sender)
        print(f"rhetorical: {v2}",flush=True)
        self.simple_retrieval_client.save_text_content(owner=self.owner,title="",content=v2,url="rhetorical",auto_chunking=False)
        return True, None 
                
        
                    