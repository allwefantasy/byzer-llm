from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union,Annotated
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
from byzerllm.apps.agent.extensions.simple_retrieval_client import SimpleRetrievalClient
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter,Document
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    
class SparkSQLAgent(ConversableAgent): 
    DEFAULT_SYSTEM_MESSAGE='''You are a helpful AI assistant. You are also a Spark SQL expert. 
你总是对问题进行拆解，先给出详细解决问题的思路，最后确保你生成的代码都在一个 SQL Block里。特别需要注意的事，你生成的Block需要用sql标注而非vbnet'''
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

        # check if the user's question is ambiguous or not, if it is, try to ask the user to clarify the question. 
        flag = [None]      
            
        def reply_with_clarify(content:Annotated[str,"这个是你反问用户的内容"]):
            '''
            如果你不理解用户的问题，那么你可以调用这个函数，来反问用户。
            '''
            flag[0] = content            
    
        last_conversation = [{"role":"user","content":'''首先先回答，你有什么不理解的地方么？比如问题如果有歧义，或者问题的信息不够，导致你无法回答。
如果有，请不要生成代码，用中文询问我，给我可能的解决方案，并且调用相应的函数。如果没有什么不理解的地方，则不需要调用任何函数。'''}]        
        self.llm.chat_oai(conversations=messages + last_conversation,
                          tools=[reply_with_clarify],
                          execute_tool=True)
        
        if flag[0] is not None:            
            return True,{"content":flag[0],"metadata":{"TERMINATE":True}}
                
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
            codes = code_utils.extract_code(conversation["content"])
            sql_codes = code_utils.get_target_codes(codes,["sql"])
            
            if sql_codes:
                reply = self.execute_spark_sql(sql_codes[0])
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

        def reply_with_review_success(sql:Annotated[str,"我们刚刚给到reviewer的sql代码"]):
            '''
            如果reviewer觉得代码没问题，那么可以调用该函数
            '''            
            target_message["content"] = messages[-2]["content"]

        def reply_with_review_fail(content:Annotated[str,"根据用户反馈的问题，我们修正后的SQL代码"]):
            '''
            如果reviewer觉得代码还有问题，那么可以调用该函数
            '''
            target_message["content"] = content
            target_message["metadata"]["TERMINATE"] = False    


        self.llm.chat_oai(conversations=messages,
                          tools=[reply_with_review_success,reply_with_review_fail],
                          execute_tool=True)                

        ## make sure the last message is the reviewed sql code    
        return True, target_message
    
    def execute_spark_sql(self,sql:Annotated[str,"Spark SQL 语句"])->str:
        '''
        执行 Spark SQL 语句
        '''
        v = self.llm._rest_byzer_script(sql,url=self.byzer_url)
        return json.dumps(v)

        
    
        
        



            

        