from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union,Annotated
from ....utils.client import ByzerLLM,message_utils,code_utils
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from .. import get_agent_name,run_agent_func,ChatResponse
import json
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    
class ByzerEngineAgent(ConversableAgent): 
    DEFAULT_SYSTEM_MESSAGE='''You are a helpful AI assistant.'''
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval, 
        chat_name:str,
        owner:str,               
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

        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], ByzerEngineAgent.generate_custom_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 

    def generate_custom_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
        ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  

        if messages is None:
            messages = self._messages[get_agent_name(sender)]

        message = messages[-1]    
        code = code_utils.extract_code(message["content"])[0][1]
        try :                            
            reply = self.execute_spark_sql(code)
        except Exception as e:
            # get full exception
            import traceback
            reply = f"执行代码出错：{traceback.format_exc()} {e}" 
            new_message = message_utils.fail_message({"content":reply} )
            print(f"Byzer Engine execute code error: {message_utils.copy_error_count(message,new_message)}",flush=True)
            return True, message_utils.copy_error_count(message,new_message)
        
        print(f"Byzer Engine execute code success: {reply}",flush=True)
        return True, message_utils.success_message({"content":reply})
    
    def execute_spark_sql(self,sql:Annotated[str,"Spark SQL 语句"])->str:
        '''
        执行 Spark SQL 语句
        '''
        
        print(f"Byzer Engine execute spark sql: {sql}",flush=True)

        v = self.llm._rest_byzer_script(f"""
load csv.`file:///home/byzerllm/projects/jupyter-workspace/nlp2query/h.csv` where header="true" as test_table;
!profiler sql '''
{sql}                                        
''';
""",owner="william",url="http://192.168.1.248:9003/run/script")
        return json.dumps(v,ensure_ascii=False)

        