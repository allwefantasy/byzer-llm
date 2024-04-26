from langchain.prompts import PromptTemplate
import dataclasses
import pydantic
from enum import Enum
from loguru import logger
from typing import Dict,Any,List,Optional,Union,Tuple,Callable,Annotated
from byzerllm.utils import (function_calling_format,
                            response_class_format,
                            response_class_format_after_chat                            
                            )

class Role:
    User = "user"
    Assistant = "assistant"
    System = "system"

@dataclasses.dataclass
class LLMHistoryItem:
      role: str
      content: str

@dataclasses.dataclass
class LLMResponse:
    output: Union[str,List[float]]
    input: Union[str,Dict[str,Any]]
    metadata: Dict[str,Any] = dataclasses.field(default_factory=dict)


class LLMFunctionCallResponse(pydantic.BaseModel):
    response:LLMResponse
    values:List[Any]
    metadata:Dict[str,Any]


class LLMClassResponse(pydantic.BaseModel):
    response:LLMResponse
    value:Optional[Any]
    metadata:Dict[str,Any]

@dataclasses.dataclass
class LLMRequest:
    instruction: Union[str,List[str]]
    embedding: bool = False
    max_length: int = 4096
    top_p: float = 0.95
    temperature: float = 0.1    
        

@dataclasses.dataclass
class FintuneRequestExtra:
    max_seq_length: int = 1024
    num_train_epochs: int = 1
    logging_steps: int = 100
    save_steps: int = 100
    extra_params: Dict[str,Any] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class  FintuneRequest:
    model_path: str
    pretrained_model_type: str
    input_data_path: str
    extra_params: FintuneRequestExtra = FintuneRequestExtra()


class InferBackend:
    Transformers = "transformers"
    VLLM = "ray/vllm"
    LLAMA_CPP = "llama_cpp"
    DeepSpeed = "ray/deepspeed"

@dataclasses.dataclass
class ExecuteCodeResponse:
      status: int
      output: str      
      code: str
      prompt: str
      variables: Dict[str,Any]=dataclasses.field(default_factory=dict)

class EventName(Enum):
    BEFORE_CALL_MODEL = "before_call_model"
    AFTER_CALL_MODEL = "after_call_model"

EventCallbackResult = Tuple[bool, Optional[Any]]
EventCallback = Callable[..., EventCallbackResult]    

class Template:
    def __init__(self,
                 role_mapping:Dict[str,str],
                 generation_config:Dict[str,Any],
                 clean_func:Callable[[str],str]=lambda s: s,
                 function_calling_format_func=function_calling_format,
                 response_class_format_func=response_class_format,
                 response_class_format_after_chat_func=response_class_format_after_chat
                 ) -> None:
        self.role_mapping = role_mapping
        self.generation_config = generation_config
        self.clean_func = clean_func        
        self.function_calling_format_func = function_calling_format_func
        self.response_class_format_func = response_class_format_func
        self.response_class_format_after_chat_func = response_class_format_after_chat_func


class Templates:

    def default_format(t,v):
        return f"{t}{v}"


    @staticmethod
    def qwen():
        def clean_func(v):            
            if "<|im_end|>" in v:
                v = v.split("<|im_end|>")[0]
            if "<|endoftext|>" in v:
                v = v.split("<|endoftext|>")[0] 
            if "<|im_start|>" in v:             
                v = v.split("<|im_start|>")[0]   
            return v   

        def sys_format(t,v):
            m = PromptTemplate.from_template(t)
            return m.format(system_msg=v)


        return Template(role_mapping={
                        "user_role":"<|im_start|>user\n",
                        "assistant_role": "<|im_end|>\n<|im_start|>assistant\n",
                        "system_msg":"<|im_start|>system\n{system_msg}<|im_end|>",
                        "system_msg_func":sys_format
                        },
                        generation_config={                            
                            "generation.repetition_penalty":1.1,
                            "generation.stop_token_ids":[151643,151645]},                  
                        clean_func=clean_func) 
    
    @staticmethod
    def llama():
        def sys_format(t,v):
            m = PromptTemplate.from_template(t)
            return m.format(system_msg=v)
        
        def user_format(t,v):
            return f"<s>[INST] {v} [/INST]"
        
        def assistant_format(t,v):
            return f" {v} </s>"
        
        return Template(
            role_mapping={
               "user_role":"",
               "assistant_role": "",
               "system_msg":"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n[/INST]</s>",
               "system_msg_func":sys_format,
               "user_role_func": user_format,
               "assistant_role_func": assistant_format
            },            
            generation_config={},
            clean_func=lambda s: s
        )
    
    @staticmethod
    def deepseek_code_chat():
        '''
        DeepSeek Coder Chat mode template:

        ### Instruction:
        ['content']
        ### Response:
        ['content']
        <|EOT|>
        ### Instruction:
        ['content']
        ### Response:
        '''
        

        def sys_format(t:Annotated[str,"the field system_msg in role_mapping "],
                       v:Annotated[str,"the system message in chat"]):
            m = PromptTemplate.from_template(t)
            return m.format(system_msg=v)
        
        def user_format(t:Annotated[str,"the field user_role in role_mapping"],
                        v:Annotated[str,"the user message in chat"]):
            '''
            format single user message
            '''
            return f"### Instruction:\n{v}"
        
        def assistant_format(t:Annotated[str,"the field assistant_role in role_mapping"],
                             v:Annotated[str,"the assistant message in chat"]):
            '''
            format single assitant message.
            
            Notice that here we do not use `t` , because we will
            use the `t` as the final suffix.
            '''
            return f"### Response:\n{v}\n<|EOT|>"
        
        return Template(
            role_mapping={
               "user_role":"",
               "assistant_role": "### Response:\n",
               "system_msg":"{system_msg}",
               "system_msg_func":sys_format,
               "user_role_func": user_format,
               "assistant_role_func": assistant_format
            },            
            generation_config={"generation.stop_token_ids":[32021]},
            clean_func=lambda s: s
        )
    @staticmethod
    def deepseek_code_insertion():        
        def sys_format(t,v):
            if "<｜fim▁hole｜>" not in v:
                raise Exception("the system message should contains <｜fim▁hole｜>")
            m = PromptTemplate.from_template(t)
            return m.format(system_msg=v)
        
        def user_format(t,v):            
            return ""
        
        def assistant_format(t,v):            
            return ""
        
        return Template(
            role_mapping={
               "user_role":"",
               "assistant_role": "",
               "system_msg":"<｜fim▁begin｜>{system_msg}<｜fim▁end｜>",
               "system_msg_func":sys_format,
               "user_role_func": user_format,
               "assistant_role_func": assistant_format
            },            
            generation_config={},
            clean_func=lambda s: s
        )
    
    @staticmethod
    def deepseek_code_completion():        
        def sys_format(t,v):            
            m = PromptTemplate.from_template(t)
            return m.format(system_msg=v)
        
        def user_format(t,v):            
            return ""
        
        def assistant_format(t,v):            
            return ""
        
        return Template(
            role_mapping={
               "user_role":"",
               "assistant_role": "",
               "system_msg":"{system_msg}",
               "system_msg_func":sys_format,
               "user_role_func": user_format,
               "assistant_role_func": assistant_format
            },            
            generation_config={},
            clean_func=lambda s: s
        )
    @staticmethod
    def yi():
        def clean_func(v):                    
            return v   

        def sys_format(t,v):
            m = PromptTemplate.from_template(t)
            return m.format(system_msg=v)


        return Template(role_mapping={
                        "user_role":"<|im_start|>user\n",
                        "assistant_role": "<|im_end|>\n<|im_start|>assistant\n",
                        "system_msg":"<|im_start|>system\n{system_msg}<|im_end|>",
                        "system_msg_func":sys_format
                        },
                        generation_config={"generation.stop_token_ids":[7]},                  
                        clean_func=clean_func) 

    @staticmethod
    def default():
        def clean_func(v):                    
            return v   

        def sys_format(t,v):
            return v

        return Template(role_mapping={
                        "user_role":"User:",
                        "assistant_role": "Assistant:",
                        "system_msg":"You are a helpful assistant. Think it over and answer the user question correctly.",
                        "system_msg_func":sys_format
                        },
                        generation_config={},                  
                        clean_func=clean_func)   

    @staticmethod
    def empty():
        def clean_func(v):                    
            return v   

        def sys_format(t,v):
            return v

        return Template(role_mapping={
                        "user_role":"",
                        "assistant_role": "",
                        "system_msg":"",
                        "system_msg_func":sys_format
                        },
                        generation_config={},                  
                        clean_func=clean_func)

class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"  

DEFAULT_CONTEXT_WINDOW = 8192  # tokens
DEFAULT_NUM_OUTPUTS = 256  # tokens        

class LLMMetadata(pydantic.BaseModel):
    context_window: int = pydantic.Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "Total number of tokens the model can be input and output for one response."
        ),
    )
    num_output: int = pydantic.Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of tokens the model can output when generating a response.",
    )
    is_chat_model: bool = pydantic.Field(
        default=False,
        description=(
            "Set True if the model exposes a chat interface (i.e. can be passed a"
            " sequence of messages, rather than text), like OpenAI's"
            " /v1/chat/completions endpoint."
        ),
    )
    is_function_calling_model: bool = pydantic.Field(
        default=False,
        # SEE: https://openai.com/blog/function-calling-and-other-api-updates
        description=(
            "Set True if the model supports function calling messages, similar to"
            " OpenAI's function calling API. For example, converting 'Email Anya to"
            " see if she wants to get coffee next Friday' to a function call like"
            " `send_email(to: string, body: string)`."
        ),
    )
    model_name: str = pydantic.Field(
        default="unknown",
        description=(
            "The model's name used for logging, testing, and sanity checking. For some"
            " models this can be automatically discerned. For other models, like"
            " locally loaded models, this must be manually specified."
        ),
    )
    system_role: MessageRole = pydantic.Field(
        default=MessageRole.SYSTEM,
        description="The role this specific LLM provider"
        "expects for system prompt. E.g. 'SYSTEM' for OpenAI, 'CHATBOT' for Cohere",
    )  
    class Config:
        protected_namespaces = ()   