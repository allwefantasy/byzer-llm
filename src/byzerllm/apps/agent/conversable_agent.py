import asyncio
from collections import defaultdict
import copy
import inspect
import time
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union,Generator
import ray
import concurrent
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from byzerllm.utils.client import message_utils
from .agent import Agent
from ...utils.retrieval import ByzerRetrieval
from ...utils.client import ByzerLLM,default_chat_wrapper,LLMResponse
from . import get_agent_name,run_agent_func, ChatResponse
from .store import MessageStore,Message as ChatStoreMessage
from .store.stores import Stores

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x


logger = logging.getLogger(__name__)


class ConversableAgent(Agent):
    
    def __init__(
        self,
        name: str,
        llm: ByzerLLM,
        retrieval: ByzerRetrieval,                
        system_message: Optional[str] = "You are a helpful AI Assistant.",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, bool]] = None,        
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        chat_wrapper:Optional[Callable[[ByzerLLM,Optional[List[Dict]],Dict],List[LLMResponse]]] = None,
        description:str = "ConversableAgent",
        message_store: Optional[Union[str,ClientActorHandle,MessageStore]] = None,
        group_name: Optional[str] = None,
        
    ):
        super().__init__(name)
        
        self.llm = llm
        self.retrieval = retrieval   
        self.chat_wrapper = chat_wrapper
     
        self.message_store = None
        
        if message_store is not None:
            self.message_store = Stores(message_store)

        self.group_name = group_name    

        self._messages = defaultdict(list)
        self._system_message = [{"content": system_message, "role": "system"}]
        
        self._is_termination_msg = (
            is_termination_msg if is_termination_msg is not None else (
                lambda x:x.get("content", "").rstrip().endswith("TERMINATE") or x.get("metadata",{}).get("TERMINATE",False)  or x.get("content", "").rstrip().endswith("终止") or x.get("metadata",{}).get("终止",False) 
                                                                       )
        )
        self.human_input_mode = human_input_mode
        self._max_consecutive_auto_reply = (
            max_consecutive_auto_reply if max_consecutive_auto_reply is not None else -1
        )
        self._code_execution_config = {} if code_execution_config is None else code_execution_config
        self._consecutive_auto_reply_counter = defaultdict(int)
        self._max_consecutive_auto_reply_dict = defaultdict(self.max_consecutive_auto_reply)
        self._function_map = {} if function_map is None else function_map
        self._default_auto_reply = default_auto_reply
        self._reply_func_list = []
        self.reply_at_receive = defaultdict(lambda: True)
        self._agent_description = description
        
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.generate_llm_reply)           
        self.auto_register_reply()
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply)         
        
        self.stream_replies = {} 
        self.error_count = {}
        
    def get_then_increment_error_count(self,agent:Union[ClientActorHandle,Agent,str]):
        agent_name = get_agent_name(agent)
        if agent_name not in self.error_count:
            self.error_count[agent_name] = 0
        v = self.error_count[agent_name]    
        self.error_count[agent_name] += 1
        return v    

    def auto_register_reply(self):        
        for _, attr in inspect.getmembers(self, predicate=inspect.ismethod):            
            if hasattr(attr, '_is_reply'):
                self.register_reply([Agent, ClientActorHandle,str], attr) 
                
    def stream_reply(self,response_gen,**kwargs):                
        id = str(uuid.uuid4())        
        def gen(): 
            t = ""           
            for response in response_gen:
                t += response
                yield (t,None)

        self._put_stream_reply(id,gen())
        return True, {
            "content":id,
            "metadata":{"agent":self.name,"TERMINATE":True,"stream":True,"stream_id":id,**kwargs}
        }                   

    def _put_stream_reply(self,id:str,reply:Generator): 
        self.stream_replies[id] = reply

    def _stream_get_message_from_self(self,id:str):
        for item in self.stream_replies.get(id,[]):
            yield item      

    def stream_get_message(self,agent:Union[ClientActorHandle,Agent,str],id:str):
        '''
        get stream reply from agent
        '''
        if isinstance(agent,Agent):
            return agent._stream_get_message_from_self(id)
        elif isinstance(agent,str):
            t = ray.get_actor(agent) 
            return t._stream_get_message_from_self.remote(id)
        else:
            return agent._stream_get_message_from_self.remote(id)
    
    def get_name(self) -> str:
        return self._name    

    def get_function_map(self):
        """Get the function map."""
        return self._function_map
    
    def get_agent_description(self):
        return self._agent_description
    
    def update_agent_description(self,description:str):
        self._agent_description = description                       
    
    def register_reply(
        self,
        trigger: Union[Type[Agent], str, Agent, Callable[[Agent], bool], List],
        reply_func: Callable,
        position: Optional[int] = 0,
        config: Optional[Any] = None,
        reset_config: Optional[Callable] = None,
    ):
        """Register a reply function.

        The reply function will be called when the trigger matches the sender.
        The function registered later will be checked earlier by default.
        To change the order, set the position to a positive integer.

        Args:
            trigger (Agent class, str, Agent instance, callable, or list): the trigger.
                - If a class is provided, the reply function will be called when the sender is an instance of the class.
                - If a string is provided, the reply function will be called when the sender's name matches the string.
                - If an agent instance is provided, the reply function will be called when the sender is the agent instance.
                - If a callable is provided, the reply function will be called when the callable returns True.
                - If a list is provided, the reply function will be called when any of the triggers in the list is activated.
                - If None is provided, the reply function will be called only when the sender is None.
                Note: Be sure to register `None` as a trigger if you would like to trigger an auto-reply function with non-empty messages and `sender=None`.
            reply_func (Callable): the reply function.
                The function takes a recipient agent, a list of messages, a sender agent and a config as input and returns a reply message.
        ```python
        def reply_func(
            recipient: ConversableAgent,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
        ) -> Union[str, Dict, None]:
        ```
            position (int): the position of the reply function in the reply function list.
                The function registered later will be checked earlier by default.
                To change the order, set the position to a positive integer.
            config (Any): the config to be passed to the reply function.
                When an agent is reset, the config will be reset to the original value.
            reset_config (Callable): the function to reset the config.
                The function returns None. Signature: ```def reset_config(config: Any)```
        """        
        self._reply_func_list.insert(
            position,
            {
                "trigger": trigger,
                "reply_func": reply_func,
                "config": copy.copy(config),
                "init_config": config,
                "reset_config": reset_config,
            },
        )

    @property
    def system_message(self):
        """Return the system message."""
        return self._system_message[0]["content"]
    
    def get_system_message(self):
        return self.system_message

    def update_system_message(self, system_message: str):
        """Update the system message.

        Args:
            system_message (str): system message for the ChatCompletion inference.
        """
        self._system_message[0]["content"] = system_message  

    def update_max_consecutive_auto_reply(self, value: int, sender: Optional[Union[Agent,ClientActorHandle,str]] = None):
        """Update the maximum number of consecutive auto replies.

        Args:
            value (int): the maximum number of consecutive auto replies.
            sender (Agent): when the sender is provided, only update the max_consecutive_auto_reply for that sender.
        """
        if sender is None:
            self._max_consecutive_auto_reply = value
            for k in self._max_consecutive_auto_reply_dict:
                self._max_consecutive_auto_reply_dict[k] = value
        else:
            self._max_consecutive_auto_reply_dict[get_agent_name(sender)] = value

    def max_consecutive_auto_reply(self, sender: Optional[Union[Agent,ClientActorHandle]] = None) -> int:
        """The maximum number of consecutive auto replies."""
        return self._max_consecutive_auto_reply if sender is None else self._max_consecutive_auto_reply_dict[get_agent_name(sender)]

    @property
    def chat_messages(self) -> Dict[Agent, List[Dict]]:
        """A dictionary of conversations from agent to list of messages."""
        return self._messages   

    def get_chat_messages(self):
        return self.chat_messages 
    
    def last_message(self, agent: Optional[Union[Agent,ClientActorHandle,str]] = None) -> Dict:
        """The last message exchanged with the agent.

        Args:
            agent (Agent): The agent in the conversation.
                If None and more than one agent's conversations are found, an error will be raised.
                If None and only one conversation is found, the last message of the only conversation will be returned.

        Returns:
            The last message exchanged with the agent.
        """
        if agent is None:
            n_conversations = len(self._messages)
            if n_conversations == 0:
                return None
            if n_conversations == 1:
                for conversation in self._messages.values():
                    return conversation[-1]
            raise ValueError("More than one conversation is found. Please specify the sender to get the last message.")
        if get_agent_name(agent) not in self._messages.keys():
            raise KeyError(
                f"The agent '{get_agent_name(agent)}' is not present in any conversation. No history available for this agent."
            )
        return self._messages[get_agent_name(agent)][-1]
    
    @staticmethod
    def _message_to_dict(message: Union[Dict, str]):
        """Convert a message to a dictionary.

        The message can be a string or a dictionary. The string will be put in the "content" field of the new dictionary.
        """
        if isinstance(message, str):
            return {"content": message}
        elif isinstance(message, dict):
            return message
        else:
            return dict(message)

    def _append_message(self, message: Union[Dict, str], role, conversation_id: Union[ClientActorHandle,Agent,str]) -> bool:
        """Append a message to the ChatCompletion conversation.

        If the message received is a string, it will be put in the "content" field of the new dictionary.
        If the message received is a dictionary but does not have any of the two fields "content" or "function_call",
            this message is not a valid ChatCompletion message.
        If only "function_call" is provided, "content" will be set to None if not provided, and the role of the message will be forced "assistant".

        Args:
            message (dict or str): message to be appended to the ChatCompletion conversation.
            role (str): role of the message, can be "assistant" or "function".
            conversation_id (Agent): id of the conversation, should be the recipient or sender.

        Returns:
            bool: whether the message is appended to the ChatCompletion conversation.
        """
        raw_message = message
        
        if isinstance(message, ChatResponse):
            message = {"content":raw_message.output,"metadata":{"raw_message":raw_message}}

        message = self._message_to_dict(message)
        # create oai message to be appended to the oai conversation that can be passed to oai directly.
        oai_message = {k: message[k] for k in ("content", "function_call", "name", "context","metadata") if k in message}
        if "content" not in oai_message:
            if "function_call" in oai_message:
                oai_message["content"] = None  # if only function_call is provided, content will be set to None.
            else:
                return False

        oai_message["role"] = "function" if message.get("role") == "function" else role
        if "function_call" in oai_message:
            oai_message["role"] = "assistant"  # only messages with role 'assistant' can have a function call.
            oai_message["function_call"] = dict(oai_message["function_call"])
        self._messages[get_agent_name(conversation_id)].append(oai_message)
        return True    


    def reset(self):
        """Reset the agent."""
        self.clear_history()
        self.reset_consecutive_auto_reply_counter()
        self.stop_reply_at_receive()
        for reply_func_tuple in self._reply_func_list:
            if reply_func_tuple["reset_config"] is not None:
                reply_func_tuple["reset_config"](reply_func_tuple["config"])
            else:
                reply_func_tuple["config"] = copy.copy(reply_func_tuple["init_config"])

    def stop_reply_at_receive(self, sender: Optional[Union[ClientActorHandle,Agent,str]]  = None):
        """Reset the reply_at_receive of the sender."""
        if sender is None:
            self.reply_at_receive.clear()
        else:
            self.reply_at_receive[get_agent_name(sender)] = False

    def reset_consecutive_auto_reply_counter(self, sender: Optional[Union[ClientActorHandle,Agent,str]]  = None):
        """Reset the consecutive_auto_reply_counter of the sender."""
        if sender is None:
            self._consecutive_auto_reply_counter.clear()
        else:
            self._consecutive_auto_reply_counter[get_agent_name(sender)] = 0

    def clear_history(self, agent: Optional[Union[ClientActorHandle,Agent,str]] = None):
        """Clear the chat history of the agent.

        Args:
            agent: the agent with whom the chat history to clear. If None, clear the chat history with all agents.
        """
        if agent is None:
            self._messages.clear()
        else:
            self._messages[get_agent_name(agent)].clear()

    
    def set_reply_at_receive(self, sender: Optional[Union[ClientActorHandle,Agent,str]] = None, value: bool = True):
        self.reply_at_receive[get_agent_name(sender)] = value 

    def get_reply_at_receive(self, sender: Optional[Union[ClientActorHandle,Agent,str]] = None):
        return self.reply_at_receive[get_agent_name(sender)] if sender is not None else self.reply_at_receive            

    def _prepare_chat(self, recipient, clear_history):            
            self.reset_consecutive_auto_reply_counter(recipient)

            # recipient.reset_consecutive_auto_reply_counter(self)
            run_agent_func(recipient, "reset_consecutive_auto_reply_counter", self)
                        
            # recipient.reply_at_receive[self] = True 
            run_agent_func(recipient, "set_reply_at_receive", self, True)
            self.reply_at_receive[get_agent_name(recipient)] = True
            if clear_history:
                self.clear_history(recipient)
                # recipient.clear_history(self)
                run_agent_func(recipient, "clear_history", self)

    def generate_init_message(self, **context) -> Union[str, Dict]:
        """Generate the initial message for the agent.

        Override this function to customize the initial message based on user's request.
        If not overriden, "message" needs to be provided in the context.
        """
        return context["message"]            

    def initiate_chat(
        self,
        recipient: Union[ClientActorHandle,Agent,str],
        clear_history: Optional[bool] = True,
        silent: Optional[bool] = False,
        **context,
    ): 
        self._prepare_chat(recipient, clear_history)
        self.send(self.generate_init_message(**context), recipient, silent=silent)
    
    def send(
        self,
        message: Union[Dict, str],
        recipient: Union[ClientActorHandle,Agent,str], #"Agent"
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
        async_send: Optional[bool] = False,
    ) -> bool:
        valid = self._append_message(message, "assistant", recipient)
        if valid:
            if isinstance(recipient, Agent):
                if async_send:
                    with concurrent.ThreadPoolExecutor() as executor:
                        executor.submit(recipient.receive, message, self, request_reply, silent)                    
                else:
                    recipient.receive(message, self, request_reply, silent)
            elif isinstance(recipient, str):                
                t = ray.get_actor(recipient) 
                if async_send:
                    t.receive.remote(message, self.get_name(), request_reply, silent)   
                else:            
                    ray.get(t.receive.remote(message, self.get_name(), request_reply, silent))
            else:
                if async_send:
                    recipient.receive.remote(message, self.get_name(), request_reply, silent)
                else:
                    ray.get(recipient.receive.remote(message, self.get_name(), request_reply, silent))    

            return True        
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            ) 
        
    
    def _process_received_message(self, message, sender, silent):
            raw_message = message
            if isinstance(message, ChatResponse):                
                message = {"content":raw_message.output,"metadata":{"raw_message":raw_message}}
            message = self._message_to_dict(message)
            # When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
            valid = self._append_message(message, "user", sender)
            if not valid:
                raise ValueError(
                    "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
                )
            
            if not silent:  
                if self.message_store is not None:
                    self.message_store.put(ChatStoreMessage(
                        id=self.group_name if self.group_name is not None else "default",
                        m=message,
                        sender=get_agent_name(sender),
                        receiver=self.get_name(),
                        timestamp=time.monotonic()
                    ))
                                  
                print(colored(get_agent_name(sender), "yellow"), "(to", f"{self.name}):\n", flush=True)
                print(colored(f"{message['content']}", "green"), flush=True)
                print("\n", "-" * 80, flush=True, sep="")

    def receive(
        self,
        message: Union[Dict, str,ChatResponse],
        sender: Union[ClientActorHandle,Agent,str], #"Agent"
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):        
        self._process_received_message(message, sender, silent)

        if request_reply is False or request_reply is None and self.reply_at_receive[get_agent_name(sender)] is False:            
            return
        reply = self.generate_reply(raw_message=message, messages=self.chat_messages[get_agent_name(sender)], sender=sender)
                
        if reply is not None:                                    
            self.send(reply, sender, silent=silent)

    def generate_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        exclude: Optional[List[Callable]] = None,
    ) -> Union[str, Dict, None,ChatResponse]:
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)]                

        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if exclude and reply_func in exclude:
                continue
            if asyncio.coroutines.iscoroutinefunction(reply_func):
                continue
            # print(f'''{self.get_name()} generating reply for {get_agent_name(sender)} from {reply_func.__name__}''',flush=True)
            final, reply = reply_func(self, raw_message=raw_message, messages=messages, sender=sender, config=reply_func_tuple["config"])
            if final:                
                return reply
                         
        return self._default_auto_reply 
        
    def generate_llm_reply(
            self,
            raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
            config: Optional[Any] = None,
        ) -> Tuple[bool, Union[str, Dict, None]]:
            """Generate a reply using autogen.oai."""            
            if self.llm is None:
                return False, None
            if messages is None:
                messages = self._messages[get_agent_name(sender)]

            # TODO: #1143 handle token limit exceeded error  
            # padding the messages to user/assistant pair
            # [{'content': '', 'role': 'assistant'},{'content': '', 'role': 'user'}, {'content': '', 'role': 'assistant'},{'content': '', 'role': 'assistant'}]    
            # should be converted to
            # [{'content': '', 'role': 'user'},{'content': '', 'role': 'assistant'},{'content': '', 'role': 'user'}, {'content': '', 'role': 'assistant'},{'content': '', 'role': 'user'},{'content': '', 'role': 'assistant'},{'content': '', 'role': 'user'}]                
            temp_messages = message_utils.padding_messages_merge(messages)

            if self.chat_wrapper is None:
                response = self.llm.chat_oai(conversations=self._system_message + temp_messages,llm_config={
                    "temperature":0.1,"top_p":0.95
                })
                return True, response[0].output
                        
            response = self.chat_wrapper(self.llm,self._system_message + temp_messages)
            return True, response[0].output    

    def get_human_input(self, prompt: str) -> str:
            """Get human input.

            Override this method to customize the way to get human input.

            Args:
                prompt (str): prompt for the human input.

            Returns:
                str: human input.
            """
            reply = input(prompt)
            return reply
    
    def check_termination_and_human_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:        

        """Check if the conversation should be terminated, and if human reply is provided."""
        if config is None:
            config = self
        if messages is None:
            messages = self._messages[get_agent_name(sender)]
        message = messages[-1]
        reply = ""
        no_human_input_msg = ""
        if self.human_input_mode == "ALWAYS":
            reply = self.get_human_input(
                f"Provide feedback to {get_agent_name(sender)}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: "
            )
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # if the human input is empty, and the message is a termination message, then we will terminate the conversation
            reply = reply if reply or not self._is_termination_msg(message) else "exit"
        else:            
            if self._max_consecutive_auto_reply_dict[get_agent_name(sender)] != -1 and self._consecutive_auto_reply_counter[get_agent_name(sender)] >= self._max_consecutive_auto_reply_dict[get_agent_name(sender)]:
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    terminate = self._is_termination_msg(message)
                    reply = self.get_human_input(
                        f"Please give feedback to {get_agent_name(sender)}. Press enter or type 'exit' to stop the conversation: "
                        if terminate
                        else f"Please give feedback to {get_agent_name(sender)}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply if reply or not terminate else "exit"
            elif self._is_termination_msg(message):
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    reply = self.get_human_input(
                        f"Please give feedback to {sender.name}. Press enter or type 'exit' to stop the conversation: "
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply or "exit"

        # print the no_human_input_msg
        if no_human_input_msg:
            print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # stop the conversation
        if reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[get_agent_name(sender)] = 0
            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[get_agent_name(sender)] == 0:
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[get_agent_name(sender)] = 0
            return True, reply

        # increment the consecutive_auto_reply_counter        
        self._consecutive_auto_reply_counter[get_agent_name(sender)] += 1
        if self.human_input_mode != "NEVER":
            print(colored("\n>>>>>>>> USING AUTO REPLY...", "red"), flush=True)

        return False, None

    
                
    