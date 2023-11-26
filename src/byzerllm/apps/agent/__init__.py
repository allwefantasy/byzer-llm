
from dataclasses import dataclass
from typing import TYPE_CHECKING,Dict, List, Optional, Union
from .agent import Agent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import ray

if TYPE_CHECKING:
    from .conversable_agent import ConversableAgent

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