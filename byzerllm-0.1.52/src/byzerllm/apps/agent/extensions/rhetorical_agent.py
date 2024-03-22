from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,message_utils,code_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from .. import get_agent_name,run_agent_func,ChatResponse
from byzerllm.apps.agent.extensions.simple_retrieval_client import SimpleRetrievalClient
from byzerllm.utils.retrieval import SearchQuery
import json
   
class RhetoricalAgent(ConversableAgent): 
    
    DEFAULT_SYSTEM_MESSAGE = '''你的主要工作是从我们的对话中找到新的名词定义

你首先需要了解 "名词定义"的含义：

所谓名词定义是在我们的对话中，有一些名词，是在交互过程中确定的，比如：

```
用户： 上分的销售额是多少？
助手： 你是指上海分公司的销售额吗？
用户： 是上海分行的销售额。
```

在这个对话中，我们可以确定，用户说的“上分”，就是指“上海分行”。于是，根据前面分析，你得到一个新的名词定义：

```json
[
  "上分是指上海分行"
]
```

再比如：

```
用户： 奔驰上个月的销量
助手： 销量是指销售额还是销售数量？
用户： 以后我说销量的时候都是指的销售额
```

在这个对话中，我们可以确定，用户以后说销售，实际上就是指销售额。于是，根据前面分析，你得到一个新的名词定义：

```json
[
 "销量是指销售额"
]
```

每次当用户说“开始”，你就可以按如下步骤进行检查：

1. 找到用户最近的一个的问题
2. 顺着这个问题，依次回顾自己的回答和用户的回复
3. 从这个过程中，参考前面的例子，找到你认为新的名词定义

请按这个步骤一步一步进行检查，并且输出每一步检查的结果。

最后，对你的结果重新以 Json 格式进行输出，格式如下：

```json
[
  "这里替换成你新发现的名词定义"
]
```

注意：

1. 当用户提供表信息或者示例数据的时候，不要对这些内容做任何分析。
2. 输出的json 需要使用 ```sql`` 进行包裹。
3. json 中的内容只需要包含名词定义部分，不要有其他内容。
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

        if self.retrieval is None:
            return True, None 
        
        ## get the last 100 conversation
        docs = self.retrieval.filter(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"user_memory",
                                         filters={"and":[{"field":"owner","value":self.owner},{"field":"chat_name","value":self.chat_name}]},
                                         sorts =[{"created_time":"desc"}],
                                        keyword=None,fields=[],
                                        vector=[],vectorField=None,
                                        limit=100)])
        docs.reverse()
        conversations = [{"content":doc["raw_content"],"role":doc["role"]} for doc in docs]
                         
        last_conversation = [{"role":"user","content":'''开始'''}]
        
        # always choose the last six messages to generate the reply
        c_messages = conversations[-7:-1]                
        _,v2 = self.generate_llm_reply(raw_message,message_utils.padding_messages_merge(self._system_message + c_messages + last_conversation),sender)
        print(f"rhetorical: {v2}",flush=True)

        try:            
            v = json.loads(code_utils.extract_code(v2)[-1][1])
            for temp in v:
                self.simple_retrieval_client.save_text_content(owner=self.owner,title="",content=temp,url="rhetorical",auto_chunking=False)
        except Exception:
            print(f"rhetorical error: {v2}",flush=True)                
        return True, None 
                
        
                    