from dataclasses import dataclass
import sys
from typing import Dict, List, Optional, Union
from .agent import Agent
from .conversable_agent import ConversableAgent
import logging
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from ...utils.client import ByzerLLM,code_utils
from byzerllm.utils.retrieval import ByzerRetrieval
import json
from . import get_agent_name, run_agent_func,ChatResponse

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

logger = logging.getLogger(__name__)


@dataclass
class GroupChat:
    """(In preview) A group chat class that contains the following data fields:
    - agents: a list of participating agents.
    - messages: a list of messages in the group chat.
    - max_round: the maximum number of rounds.
    - admin_name: the name of the admin agent if there is one. Default is "Admin".
        KeyBoardInterrupt will make the admin agent take over.
    - func_call_filter: whether to enforce function call filter. Default is True.
        When set to True and when a message is a function call suggestion,
        the next speaker will be chosen from an agent which contains the corresponding function name
        in its `function_map`.
    """

    agents: List[Union[Agent,ClientActorHandle,str]]
    messages: List[Dict]
    max_round: int = 10
    admin_name: str = "Admin"
    func_call_filter: bool = True

    @property
    def agent_names(self) -> List[str]:
        """Return the names of the agents in the group chat."""
        _agent_names = []
        for agent in self.agents:
            _agent_names.append(get_agent_name(agent))
        return _agent_names           


    def reset(self):
        """Reset the group chat."""
        self.messages.clear()

    def agent_by_name(self, name: str) -> Agent:
        """Returns the agent with a given name."""
        return self.agents[self.agent_names.index(name)]    
   

    def next_agent(self, agent: Union[Agent,ClientActorHandle,str], agents: List[Union[Agent,ClientActorHandle,str]]) -> Union[Agent,ClientActorHandle,str]:
        """Return the next agent in the list."""
        agent_name = get_agent_name(agent)
        if agents == self.agents:
            return agents[(self.agent_names.index(agent_name) + 1) % len(agents)]
        else:
            offset = self.agent_names.index(agent_name) + 1
            for i in range(len(self.agents)):
                if self.agents[(offset + i) % len(self.agents)] in agents:
                    return self.agents[(offset + i) % len(self.agents)]

    def select_speaker_msg(self, agents: List[Union[Agent,ClientActorHandle,str]]):
        """Return the message for selecting the next speaker."""
        return f"""You are in a role play game. The following roles are available:
{self._participant_roles()}.

Read the following conversation.
Then select the next role from {[get_agent_name(agent) for agent in agents]} to play. Only return the role."""
     
    

    def select_speaker(self, last_speaker: Union[Agent,ClientActorHandle,str], 
                       selector: Union[ConversableAgent,ClientActorHandle,str],):
        """Select the next speaker."""
        if self.func_call_filter and self.messages and "function_call" in self.messages[-1]:
            # find agents with the right function_map which contains the function name
            agents = [
                agent for agent in self.agents if run_agent_func(agent,"can_execute_function",self.messages[-1]["function_call"]["name"])
            ]
            if len(agents) == 1:
                # only one agent can execute the function
                return agents[0]
            elif not agents:
                # find all the agents with function_map
                agents = [agent for agent in self.agents if run_agent_func(agent,"function_map")]
                if len(agents) == 1:
                    return agents[0]
                elif not agents:
                    raise ValueError(
                        f"No agent can execute the function {self.messages[-1]['name']}. "
                        "Please check the function_map of the agents."
                    )
        else:
            agents = self.agents
            # Warn if GroupChat is underpopulated
            n_agents = len(agents)
            if n_agents < 3:
                logger.warning(
                    f"GroupChat is underpopulated with {n_agents} agents. Direct communication would be more efficient."
                )
        
        run_agent_func(selector,"update_system_message",self.select_speaker_msg(agents))

        select_prompt = self.messages    +        [
                {
                    "role": "user",
                    "content": f"Read the above conversation. Then select the next role from {[get_agent_name(agent) for agent in agents]} to play. Only return the role.",
                }
            ]        
        
        
        final, name = run_agent_func(selector,"generate_llm_reply",None,select_prompt)                                        
        print(colored(f"GroupChat select_speaker: {name}","green"))

        if not final:
            # i = self._random.randint(0, len(self._agent_names) - 1)  # randomly pick an id
            return self.next_agent(last_speaker, agents)
        try:
            return self.agent_by_name(name.strip())
        except ValueError:
            logger.warning(
                f"GroupChat select_speaker failed to resolve the next speaker's name. Speaker selection will default to the next speaker in the list. This is because the speaker selection OAI call returned:\n{name}"
            )
            return self.next_agent(last_speaker, agents)

    def _participant_roles(self):
        roles = []
        for agent in self.agents:
            if run_agent_func(agent,"get_system_message").strip() == "":
                logger.warning(
                    f"The agent '{get_agent_name(agent)}' has an empty system_message, and may not work well with GroupChat."
                )
            roles.append(f"{get_agent_name(agent)}: {run_agent_func(agent,'get_system_message')}")
        return "\n".join(roles)


class GroupChatManager(ConversableAgent):
    """(In preview) A chat manager agent that can manage a group chat of multiple agents."""

    def __init__(
        self,
        groupchat: GroupChat,
        llm:ByzerLLM,
        retrieval:ByzerRetrieval,
        name: Optional[str] = "chat_manager",        
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        system_message: Optional[str] = "Group chat manager.",
        **kwargs,
    ):
        super().__init__(
            name=name,
            llm=llm,
            retrieval=retrieval,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            system_message=system_message,
            **kwargs,
        )
        # Order of register_reply is important.
        # Allow sync chat if initiated using initiate_chat
        self.register_reply([Agent, ClientActorHandle,str], 
                            GroupChatManager.run_chat, 
                            config=groupchat, 
                            reset_config=GroupChat.reset)  
        self.groupchat = groupchat  

    def get_groupchat(self) -> GroupChat:
        """Return the group chat."""
        return self.groupchat   

    def reset_agents(self):
        """Reset the agents."""
        for agent in self.groupchat.agents:
            run_agent_func(agent,"reset")       

    def run_chat(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[Agent, ClientActorHandle,str]] = None,
        config: Optional[GroupChat] = None,
    ) -> Union[str, Dict, None]:
        """Run a group chat."""
        if messages is None:
            messages = self._messages[get_agent_name(sender)]
        message = messages[-1]
        speaker = sender
        groupchat = config
        for i in range(groupchat.max_round):
            print(colored(f"GroupChatManager run_chat: {i}","green"),flush=True)
            # set the name to speaker's name if the role is not function
            if message["role"] != "function":
                message["name"] = get_agent_name(speaker)
            groupchat.messages.append(message)
            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if get_agent_name(agent) != get_agent_name(speaker):
                    self.send(message, agent, request_reply=False, silent=True)
            if i == groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                # let the speaker speak
                reply = run_agent_func(speaker,"generate_reply",sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = run_agent_func(speaker,"generate_reply",sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            run_agent_func(speaker,"send",message=reply,recipient=get_agent_name(self),request_reply=False);
            # get the speaker's last message and in next round, broadcast it to all other agents            
            message = self.last_message(speaker)            
        return True, None
    
