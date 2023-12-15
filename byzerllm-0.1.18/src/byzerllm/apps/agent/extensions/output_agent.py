from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from .. import get_agent_name,run_agent_func,ChatResponse
from langchain import PromptTemplate

SYSTEM_PROMPT = '''You are a helpful AI assistant. The user will give you a conversation.
You should check the conversation is
'''
USER_PROMPT = '''User's question is: {input_question}

The conversation is: 

```
{input_context}
```
'''
class OutputAgent(ConversableAgent):
    

    def __init__(
        self,
        name: str,
        llm: ByzerLLM,        
        retrieval: ByzerRetrieval,                
        system_message: Optional[str] = SYSTEM_PROMPT,        
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
        self.register_reply([Agent, ClientActorHandle,str], OutputAgent.generate_converation_based_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply)

    def generate_retrieval_based_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  
        
        box = []

        for message in messages:
            if not message["content"]:
                continue
            if message["role"] == "user":
                question = message["content"]
                box.append("Q:" + question)
            if message["role"] == "assistant":
                context = message["content"]   
                box.append("A:" + context)

        prompt = PromptTemplate.from_template(USER_PROMPT).format(input_question=box[0],input_context="\n".join(box))        
        new_message = {"content":prompt,"role":"user"}                    
        final,v = self.generate_llm_reply(None,[new_message],sender)
        return True,v + " TERMINATE"

                
