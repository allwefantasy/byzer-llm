from ..conversable_agent import ConversableAgent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from ....utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from ..agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import time
import ray
from .. import get_agent_name,run_agent_func,ChatResponse,modify_message_metadata,modify_message_content
from langchain import PromptTemplate
from byzerllm.utils.client import default_chat_wrapper,LLMResponse
from byzerllm.apps.agent import Agents
from byzerllm.apps.agent.extensions.preview_file_agent import PreviewFileAgent
from byzerllm.apps.agent.extensions.python_codesandbox_agent import PythonSandboxAgent
from byzerllm.apps.agent.extensions.visualization_agent import VisualizationAgent
from byzerllm.apps.agent.user_proxy_agent import UserProxyAgent
from byzerllm.apps.agent.assistant_agent import AssistantAgent
from byzerllm.utils import generate_str_md5
import os
class DataAnalysisPipeline(ConversableAgent):  
    DEFAULT_SYSTEM_MESSAGE = '''You are a helpful data analysis assistant.
You don't need to write code, or anwser the question. The only thing you need to do 
is plan the data analysis pipeline.

You have some tools like the following:

1. visualization_agent, this agent will help you to visualize the data.
2. assistant_agent, this agent will help you to analyze the data but not visualize it.

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
        file_path:str,
        file_ref:ClientObjectRef ,
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
        self.file_path = file_path
        self.file_ref = file_ref        
        self._reply_func_list = []
        # self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)   
        self.register_reply([Agent, ClientActorHandle,str], DataAnalysisPipeline.run_pipeline) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply) 

        params = {}
        if "chat_wrapper" in kwargs:
            params["chat_wrapper"] = kwargs["chat_wrapper"]        

        self.python_interpreter = Agents.create_local_agent(PythonSandboxAgent,"python_interpreter",
                                                llm,retrieval,
                                                max_consecutive_auto_reply=100,
                                                system_message="you are a code sandbox",**params
                                                )

        self.preview_file_agent = Agents.create_local_agent(PreviewFileAgent,"privew_file_agent",llm,retrieval,
                                        max_consecutive_auto_reply=100,
                                        code_agent = self.python_interpreter,**params
                                        )
        
        self.visualization_agent = Agents.create_local_agent(VisualizationAgent,"visualization_agent",llm,retrieval,
                                        max_consecutive_auto_reply=100,                                        
                                        code_agent = self.python_interpreter,**params
                                        ) 
        self.assistant_agent = Agents.create_local_agent(AssistantAgent,"assistant_agent",llm,retrieval,
                                        max_consecutive_auto_reply=100,
                                        code_agent = self.python_interpreter,**params
                                        )                 
        
        self.agents = {
            "assistant_agent":self.assistant_agent,
            "visualization_agent":self.visualization_agent
        }

    def preview_file(self):
        self.preview_file_agent._prepare_chat(self.python_interpreter, True)        
        self.max_consecutive_auto_reply = 0          
        self.initiate_chat(
        self.preview_file_agent,
        message={
            "content":f"We have a file, the file path is: {self.file_path} , please preview this file",
            "metadata":{
                "file_path":self.file_path,
                "file_ref":self.file_ref
            }
        })      
        # sync the conversation of preview_file_agent to other agents
        print("sync the conversation of preview_file_agent to other agents",flush=True)
        for agent in self.agents.values():            
            for message in self._messages["privew_file_agent"]:                 
                self.send(message=message,recipient=agent,request_reply=False)

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

        _,agent_name = self.select_agent(raw_message,messages,sender)
        print(f"Select agent: {agent_name} to answer the question: {ori_message['content'][0:20]}",flush=True)
        
        if agent_name:
            agent = self.agents[agent_name]
            # reset the agent except the conversation history  
            self._prepare_chat(agent, clear_history=False)                      
            self.send(message=ori_message,recipient=agent,request_reply=False)                                                
            agent_reply = agent.generate_reply(raw_message=None,messages=None,sender=self)
            if isinstance(agent_reply,dict):
                agent_reply = agent_reply["content"]
            return True, {"content":agent_reply,"metadata":{"TERMINATE":True}}
        
        return self.generate_llm_reply(raw_message,messages,sender)

class DataAnalysisPipelineManager:
    def __init__(self) -> None:
        self.pipelines = {}
        self.lasted_updated = {}
    
    ## if the sandbox is not used for 1h, we will remove it
    def check_pipeline_timeout(self,timeout:int=60*60): 
        remove_names = []
        for name in self.lasted_updated:
            if time.time() - self.lasted_updated[name] > timeout:
                remove_names.append(name)
        for name in remove_names:
            del self.pipelines[name]
            del self.lasted_updated[name]        

    def check_pipeline_exists(self,name:str)->bool:
        self.check_pipeline_timeout() 
        return name in self.pipelines

    def get_pipeline(self,name:str):                
        self.check_pipeline_timeout()        
        return self.pipelines[name]
    
    def force_clear(self):
        self.pipelines = {}
        self.lasted_updated = {}

    def get_or_create_pipeline( self,
                                name:str,
                                llm:ByzerLLM,retrieval:ByzerRetrieval,
                                file_path:str,
                                file_ref:ClientObjectRef,
                                chat_wrapper:Optional[Callable[[ByzerLLM,Optional[List[Dict]],Dict],List[LLMResponse]]] = default_chat_wrapper,
                                num_gpus:int=0,num_cpus:int=0):
        self.lasted_updated[name] = time.time()
        self.check_pipeline_timeout()
        if name in self.pipelines:            
            return self.pipelines[name]
        
        pipeline = ray.remote(DataAnalysisPipeline).options(
                name=name,                                              
                num_cpus=num_cpus,
                num_gpus=num_gpus
            ).remote(
                name = name,
                llm = llm,
                retrieval = retrieval,
                file_path=file_path,
                file_ref=file_ref,
                chat_wrapper=chat_wrapper
                )
        self.pipelines[name] = pipeline
        return pipeline
    
class DataAnalysis:
    def __init__(self,chat_name:str, 
                 owner:str,
                 file_path:str,
                 llm:ByzerLLM,
                 retrieval:ByzerRetrieval,
                 use_shared_disk:bool=False, 
                 chat_wrapper:Optional[Callable[[ByzerLLM,Optional[List[Dict]],Dict],List[LLMResponse]]] = default_chat_wrapper               
                 ):
        self.chat_name = chat_name
        self.owner = owner
        self.chat_wrapper = chat_wrapper
        self.suffix = generate_str_md5(f"{self.chat_name}_{self.owner}")
        self.name = f"data_analysis_pp_{self.suffix}"   
        self.manager = self.get_pipeline_manager()  
        self.file_path = file_path
        self.use_shared_disk = use_shared_disk
        self.llm = llm
        self.retrieval = retrieval
        
        if not ray.get(self.manager.check_pipeline_exists.remote(self.name)):
            if self.file_path and not self.use_shared_disk:
                base_name = os.path.basename(file_path)
                _, ext = os.path.splitext(base_name)
                new_base_name = self.name + ext
                dir_name = os.path.dirname(file_path)
                new_file_path = os.path.join(dir_name, new_base_name)
                print(f"use_shared_disk: {self.use_shared_disk} file_path: {self.file_path} new_file_path: {new_file_path}",flush=True)
                self.file_ref = ray.put(open(self.file_path,"rb").read())
                self.file_path = new_file_path

            self.data_analysis_pipeline = ray.get(self.manager.get_or_create_pipeline.remote(
                name = self.name,
                llm =llm,
                retrieval =retrieval,
                file_path=self.file_path,
                file_ref=self.file_ref,
                chat_wrapper = self.chat_wrapper
                )) 

            # trigger file preview manually
            ray.get(self.data_analysis_pipeline.preview_file.remote()) 
        else:
            self.data_analysis_pipeline = ray.get(self.manager.get_pipeline.remote(self.name))

        self.client = self.get_or_create_user(f"user_{self.name}")

    def get_or_create_user(self,name:str)->bool:
        try:
            return ray.get_actor(name)            
        except Exception:
            return Agents.create_remote_agent(UserProxyAgent,f"user_{self.name}",self.llm,self.retrieval,
                                human_input_mode="NEVER",
                                max_consecutive_auto_reply=0,chat_wrapper=self.chat_wrapper)
        
    def analyze(self,content:str):        
        ray.get(self.data_analysis_pipeline.update_max_consecutive_auto_reply.remote(1))
        ray.get(           
           self.client.initiate_chat.remote(
                self.data_analysis_pipeline,
                message={
                    "content":content,
                    "metadata":{                        
                    }
                },
           ) 
        )
        # o = str(self.output())
        # # remove \nTERMINATE from the output, recursively since the output may contain multi \nTERMINATE in the end
        # while o[-10:] == "\nTERMINATE":
        #     o = o[:-10]
        
        return self.output()    
        
    
    def output(self):
        return ray.get(self.data_analysis_pipeline.last_message.remote(get_agent_name(self.client)))        
    
    def get_pipeline_manager(self)->ClientActorHandle:
        name = "DATA_ANALYSIS_PIPELINE_MANAGER"
        manager = None
        try:
            manager = ray.get_actor(name)
        except Exception:              
            manager = ray.remote(DataAnalysisPipelineManager).options(
                name=name, 
                lifetime="detached", 
                max_concurrency=500,              
                num_cpus=1,
                num_gpus=0
            ).remote()
        return manager     
        
        
    
        