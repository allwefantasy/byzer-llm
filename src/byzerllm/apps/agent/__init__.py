
from dataclasses import dataclass
from typing import TYPE_CHECKING,Dict, List, Optional, Union,Any
from .agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import ray
import dataclasses

if TYPE_CHECKING:
    from .conversable_agent import ConversableAgent

@dataclasses.dataclass
class ChatResponse:
      status: int
      output: str      
      code: str
      prompt: str
      variables: Dict[str,Any]=dataclasses.field(default_factory=dict)

def get_agent_name(agent: Union[Agent,ClientActorHandle,str]) -> str:
    if isinstance(agent,Agent):
        agent_name = agent.name
    elif isinstance(agent,str):
        agent_name = agent
    else:    
        agent_name = ray.get(agent.get_name.remote())

    return agent_name    

def run_agent_func(agent: Union[Agent,"ConversableAgent",ClientActorHandle], func_name: str, *args, **kwargs):
    """Run a function of an agent."""
    if isinstance(agent,Agent):
        return getattr(agent, func_name)(*args, **kwargs)
    elif isinstance(agent,str):
        return ray.get(getattr(ray.get_actor(agent), func_name).remote(*args, **kwargs))    
    else:
        return ray.get(getattr(agent, func_name).remote(*args, **kwargs)) 

class Agents:
    @staticmethod
    def create_remote_agent(cls,name:str,llm,retrieval,*args, **kwargs)->ClientActorHandle:
        return ray.remote(name=name,max_concurrency=10)(cls).remote(
        name=name,llm=llm,retrieval=retrieval,*args, **kwargs)   

    @staticmethod
    def group(group_name:str,agents: List[Union[Agent,ClientActorHandle,str]],llm,retrieval,*args, **kwargs) -> List[Union[Agent,ClientActorHandle,str]]:
        from .groupchat import GroupChat
        from .groupchat import GroupChatManager
        
        groupchat = GroupChat(agents=agents, *args, **kwargs)
        group_chat_manager =Agents.create_remote_agent(GroupChatManager,name=group_name,llm=llm,retrieval=retrieval,groupchat=groupchat,*args, **kwargs)
        return group_chat_manager   
    
    

        