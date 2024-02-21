
import ray
import os

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from byzerllm.utils import generate_str_md5
from byzerllm.utils.client import ByzerLLM,default_chat_wrapper,LLMResponse
from byzerllm.utils.retrieval import ByzerRetrieval
from ray.util.client.common import ClientActorHandle, ClientObjectRef

from byzerllm.apps.agent import Agent,Agents,get_agent_name,run_agent_func,ChatResponse,modify_message_metadata,modify_message_content

from byzerllm.apps.agent.user_proxy_agent import UserProxyAgent
from byzerllm.apps.agent.extensions.data_analysis_pipeline_agent import DataAnalysisPipeline,DataAnalysisPipelineManager
from byzerllm.apps.agent.extensions.simple_retrieval_client import SimpleRetrievalClient
from byzerllm.apps.agent.store.memory_store import  MessageStore,MemoryStore,Message as ChatStoreMessage
from byzerllm.apps.agent.store.stores import Stores
try:
    from termcolor import colored
except ImportError:
    def colored(x, *args, **kwargs):
        return x


class DataAnalysis:
    def __init__(self,chat_name:str, 
                 owner:str,
                 file_path:str,
                 llm:ByzerLLM,
                 retrieval:ByzerRetrieval,
                 use_shared_disk:bool=False, 
                 chat_wrapper = default_chat_wrapper ,
                 skip_preview_file:bool=False,
                 retrieval_cluster:str="data_analysis",
                 retrieval_db:str="data_analysis",
                 message_store:Optional[Optional[Union[str,ClientActorHandle,MessageStore]]]=None,              
                 ):
        self.chat_name = chat_name
        self.owner = owner
        self.chat_wrapper = chat_wrapper
        self.suffix = generate_str_md5(f"{self.chat_name}_{self.owner}")
        self.name = f"data_analysis_pp_{self.suffix}"   
        self.manager = self.get_pipeline_manager() 

        self.message_store = message_store 
        
        self.use_shared_disk = use_shared_disk
        self.llm = llm
        self.retrieval = retrieval
        
        self.retrieval_cluster = retrieval_cluster
        self.retrieval_db = retrieval_db

        self.simple_retrieval_client = SimpleRetrievalClient(llm=self.llm,
                                                        retrieval=self.retrieval,
                                                        retrieval_cluster=self.retrieval_cluster,
                                                        retrieval_db=self.retrieval_db,
                                                        )     
        self.file_path = file_path
        self.file_ref = None   

        if self.message_store:
            if isinstance(self.message_store,str):
                try:
                    ray.get_actor(self.message_store)
                except:
                    ray.remote(MemoryStore).options(num_cpus=0.1,name=self.message_store, lifetime="detached").remote()


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
                chat_name = self.chat_name,
                owner = self.owner,
                chat_wrapper = self.chat_wrapper,
                message_store = self.message_store,
                )) 

            # trigger file preview manually
            if not skip_preview_file:
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
        
    def get_messages(self):
        v = [] 
        store = Stores("MESSAGE_STORE")
        for item in store.get(self.name):            
            v.append(item.sender + " (to " + f"{item.receiver}):\n")
            v.append(f"{item.m['content']}")
            v.append("\n" + "-" * 80)
        return "\n".join(v)    

    def send_from_agent_to_agent(self,from_agent_name:str,to_agent_name:str,message:Dict[str,Any]):
        if self.data_analysis_pipeline is None:
            return None
        return ray.get(self.data_analysis_pipeline.send_from_agent_to_agent.remote(from_agent_name,to_agent_name,message))  
        
    def analyze(self,content:str,metadata:Dict[str,Any]={}): 
        if self.message_store is not None:
            store = Stores("MESSAGE_STORE")
            store.clear(self.name)

        ray.get(self.data_analysis_pipeline.update_max_consecutive_auto_reply.remote(1))
        ray.get(           
           self.client.initiate_chat.remote(
                self.data_analysis_pipeline,
                message={
                    "content":content,
                    "role":"user",
                    "metadata":{  
                        **metadata                      
                    }                    
                },
           ) 
        ) 
        output = self.output()   
        
        # if self.message_store is not None:
        #     self.message_store.put(ChatStoreMessage(self.name,None,None,None,-1))
        metadata = output["metadata"]        
        
        if "stream" in metadata and metadata["stream"]:
            agent = metadata["agent"]
            stream_id = metadata["stream_id"]
            result = []
            yield metadata["contexts"]
            for item in self.data_analysis_pipeline.get_agent_stream_messages.remote(agent,stream_id):
                t = ray.get(item)
                result.append(t[0])
                yield t
            
            if self.retrieval:
                self.simple_retrieval_client.save_conversation(owner=self.owner,chat_name=self.chat_name,role="user",content=content)
                r = "".join(result)
                self.simple_retrieval_client.save_conversation(owner=self.owner,chat_name=self.chat_name,role="assistant",content=r) 
      
        else:    
            if self.retrieval:
                self.simple_retrieval_client.save_conversation(owner=self.owner,chat_name=self.chat_name,role="user",content=content)
                self.simple_retrieval_client.save_conversation(owner=self.owner,chat_name=self.chat_name,role="assistant",content=output["content"])
            return output

    def get_chat_messages(self):
        return ray.get(self.data_analysis_pipeline.get_chat_messages.remote())   

    def close(self):        
        try:
            ray.kill(ray.get_actor(f"user_{self.name}"))
        except Exception:
            pass
        
        try:            
            ray.kill(ray.get_actor(f"{self.name}"))
        except Exception:
            pass

        try:            
            ray.get(self.manager.remove_pipeline.remote(self.name))  
        except Exception:
            pass

        self.data_analysis_pipeline = None                      
    
    def output(self):
        return ray.get(self.data_analysis_pipeline.last_message.remote(get_agent_name(self.client)))  

    def update_pipeline_system_message(self,system_message:str)->bool: 
        if self.data_analysis_pipeline is None:
            return False           
        ray.get(self.data_analysis_pipeline.update_system_message.remote(system_message))
        return True

    def update_agent_system_message(self,agent_name:str,system_message:str)->bool:
        if self.data_analysis_pipeline is None:
            return False 
        return ray.get(self.data_analysis_pipeline.update_system_message_by_agent.remote(agent_name,system_message))        
    
    def get_agent_system_message(self,agent_name:str)->str:
        if self.data_analysis_pipeline is None:
            return ""
        return ray.get(self.data_analysis_pipeline.get_agent_system_message.remote(agent_name))
    
    def get_pipeline_system_message(self)->str:
        if self.data_analysis_pipeline is None:
            return ""
        return ray.get(self.data_analysis_pipeline.get_system_message.remote())
    
    def get_agent_names(self):
        if self.data_analysis_pipeline is None:
            return []
        return ray.get(self.data_analysis_pipeline.get_agent_names.remote())
    
    def clear_agent_message_box(self,agent_name:str,last_n=0)->bool:
        if self.data_analysis_pipeline is None:
            return False
        return ray.get(self.data_analysis_pipeline.clear_agent_message_box.remote(agent_name,last_n))

    
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
        
        
    
        