from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM,ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
from .. import get_agent_name,run_agent_func,ChatResponse,modify_message_metadata,modify_message_content
from langchain import PromptTemplate
from byzerllm.apps.agent import Agents
from byzerllm.apps.agent.extensions.preview_file_agent import PreviewFileAgent
from byzerllm.apps.agent.extensions.python_codesandbox_agent import PythonSandboxAgent
from byzerllm.apps.agent.extensions.visualization_agent import VisualizationAgent


class DataAnalysisPipeline(ConversableAgent):  
    DEFAULT_SYSTEM_MESSAGE = '''You are a helpful data anaylisys AI assistant.
You don't need to write code, or anwser the question. The only thing you need to do 
is plan the data analysis pipeline.

You have some tools like the following:

1. visualization_agent, this agent will help you to visualize the data.

Please check the user's question and decide which tool you need to use. And then reply the tool name only.
If there is no tool can help you, 
you should reply exactly `UPDATE CONTEXT`.
''' 

    DEFAULT_USER_MESSAGE = """
"""

    def __init__(
        self,
        name: str,
        llm: ByzerLLM,
        retrieval: ByzerRetrieval,        
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
        self.register_reply([Agent, ClientActorHandle,str], DataAnalysisPipeline.run_pipeline) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 

        

        python_interpreter = Agents.create_local_agent(PythonSandboxAgent,"python_interpreter",
                                                llm,retrieval,
                                                max_consecutive_auto_reply=3,
                                                system_message="you are a code sandbox")

        preview_file_agent = Agents.create_local_agent(PreviewFileAgent,"privew_file_agent",llm,retrieval,
                                        max_consecutive_auto_reply=3,
                                        code_agent = python_interpreter
                                        )
        
        visualization_agent = Agents.create_local_agent(VisualizationAgent,"visualization_agent",llm,retrieval,
                                        max_consecutive_auto_reply=3,
                                        code_agent = python_interpreter
                                        )
        self.agents = {
            "preview_file_agent":preview_file_agent,
            "visualization_agent":visualization_agent
        }

    def select_agent(self,raw_message,messages,sender):
        _,llm_reply = self.generate_llm_reply(raw_message,messages,sender)
        fail = "UPDATE CONTEXT" in llm_reply[-20:].upper() or "UPDATE CONTEXT" in llm_reply[:20].upper()
        if fail:
            return True, None
        else:
            return True,llm_reply.strip()

    def run_pipeline(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:  
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)]
        
        ori_message = messages[-1]
        

        if "metadata" not in ori_message or "file_path" in ori_message["metadata"]:
            raise ValueError("metadata/file_path is not in the message")

        # we always need to preview file
        preview_file_agent = self.agents["preview_file_agent"]
        
        self.send(message=ori_message,recipient=preview_file_agent,request_reply=False)
        
        _,file_preview = preview_file_agent.generate_reply(None,None,self)
        

        _,agent_name = self.select_agent(raw_message,messages,sender)

        if agent_name:
            agent = self.agents[agent_name]
            
            temp_message = modify_message_metadata(ori_message,file_preview=file_preview)
            self.send(message=temp_message,recipient=agent,request_reply=False)                                                
            _,agent_reply = agent.generate_reply(raw_message=None,messages=None,sender=self)
            return True, agent_reply + "\nTERMINATE"
        
        return self.generate_llm_reply(raw_message,messages,sender)

        