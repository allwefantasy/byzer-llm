from pathlib import Path
from functools import wraps
import time
import json
from transformers import PreTrainedTokenizer,StoppingCriteria
import torch
import hashlib
import threading
from typing import TYPE_CHECKING,TypeVar,Dict, List, Optional, Union,Any,get_type_hints,Annotated,get_args,Callable
import typing
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import inspect
import pydantic

T = TypeVar("T")

def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

import signal
from contextlib import contextmanager
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

@contextmanager
def timeout(duration: float):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

def timeit(func):
    """
    Decorator to time a function.
    """

    @wraps(func)
    def inner(*args, **kwargs):
        start_time = time.monotonic()
        ret = func(*args, **kwargs)
        time_taken = time.monotonic() - start_time
        print(f"{func} took {time_taken} s to complete",flush=True)
        return ret

    return inner

def generate_instruction_from_history(ins:str,his:List[Dict[str,str]],role_mapping:Dict[str,str]={        
        "user":"User",        
        "assistant":"Assistant",
    }):

    new_his = []    
    for item in his:
        if item["role"] == "system":
            new_his.append(item["content"])
            continue        
        new_his.append(f"{role_mapping[item['role']]}:{item['content']}")            

    # here we should make sure the user build the conversation string manually also
    # works. This means if the user do not provide  the history, then
    # we should treat ins as conversation string which the user build manually
    if len(new_his) > 0 and ins != "":
        new_his.append(f"{role_mapping['user']}:{ins}")
        new_his.append(f"{role_mapping['assistant']}:")

    if len(new_his) > 0 and ins == "":
        new_his.append(f"{role_mapping['assistant']}:")            
    
    if len(new_his) == 0:
        new_his.append(ins)    

    fin_ins = "\n".join(new_his)
    return fin_ins  

def compute_max_new_tokens(tokens,max_length:int):
    input_length = tokens["input_ids"].shape[1]
    max_new_tokens = max_length - input_length
    if max_new_tokens <= 0:
        raise Exception(f"Input is too long ({input_length}). Try to reduce the length of history or use a larger `max_length` value (now:{max_length})")
    return max_new_tokens

def tokenize_string(tokenizer: PreTrainedTokenizer, key: str) -> Union[int, List[int]]:
    """Tokenize a string using a tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        key (str): String to tokenize.
    """
    token_ids = tokenizer.encode(key, add_special_tokens=False)
    return token_ids

def tokenize_stopping_sequences_where_needed(
    tokenizer: PreTrainedTokenizer,
    stopping_sequences: List[Union[str, int, List[int]]],
) -> List[Union[List[int], int]]:
    """If any sequence is a string, tokenize it.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        stopping_sequences (List[Union[str, int, List[int]]]): Stopping sequences to
            tokenize. Can be ids, sequences of ids or strings.
    """
    if not stopping_sequences:
        return None
    return [
        tokenize_string(tokenizer, sequence) if isinstance(sequence, str) else sequence
        for sequence in stopping_sequences
    ]

def  tokenize_stopping_sequences(tokenizer,stop_words):
    stop_words_ids = []
    for stop_word in stop_words:
        w = tokenize_string(tokenizer, stop_word)
        # remove the first token which is empty token 
        # this should work for only llama model
        # if w[0] == 29871 and tokenizer.decode([w[0]],skip_special_tokens=False) == "":
        #     w = w[1:]
        stop_words_ids.append(w)    
    return stop_words_ids

class StopSequencesCriteria(StoppingCriteria):
    """
     skip_check_min_length is used to skip the the stop sequence check if the input_ids is short
     than the min_length. 
    """
    def __init__(self, tokenizer,stops = [],input_start=0, skip_check_min_length=0):
    
      super().__init__()      
      self.stops = stops
      self.input_start = input_start
      self.skip_check_min_length = skip_check_min_length
      self.stop_words= [tokenizer.decode(item,skip_special_tokens=True) for item in stops]
      self.tokenizer = tokenizer   

    def to_str(self,s):
        return self.tokenizer.decode(s,skip_special_tokens=True)     

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):                   
      for index,stop in enumerate(self.stops):                        
        if  self.to_str(input_ids[0][-(len(stop)+10):]).endswith(self.stop_words[index]):
            return True
      return False

def load_json_str(json_str:str):        
    return json.loads(json_str) 


def generate_file_md5(file_path: str) -> str:
    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def generate_str_md5(s: str) -> str:
    md5_hash = hashlib.md5()
    md5_hash.update(s.encode("utf-8"))
    return md5_hash.hexdigest() 

class SingleOutput:
    def __init__(self, text:str):
        self.text = text
        
class StreamOutputs: 
    def __init__(self, outputs:List[SingleOutput]):
        self.outputs = outputs        

class BlockVLLMStreamServer:
    def __init__(self):
        self.cache = {}
        self.cache_status = {} 
        self.lock = threading.Lock()

    def add_item(self, request_id, item):
        with self.lock:            
            self.cache[request_id]=item
            self.cache_status[request_id]=int(time.time()*1000)
    
    def mark_done(self, request_id):
        if len(self.cache_status) > 30:
            now = int(time.time()*1000)
            with self.lock:
                for k in list(self.cache_status.keys()):
                    if now - self.cache_status[k] > 10*60*60*1000:
                        del self.cache_status[k]
                        del self.cache[k] 
        with self.lock:            
            self.cache_status[request_id] = 0

    def get_item(self, request_id):                
        with self.lock:
            v = self.cache.get(request_id, None)     
            if request_id in self.cache_status and self.cache_status[request_id] == 0:
                del self.cache[request_id]
                del self.cache_status[request_id]
            return v     

class VLLMStreamServer:
    def __init__(self):
        self.cache = {}
        self.cache_status = {} 
        self.lock = threading.Lock()

    async def add_item(self, request_id, item):
        with self.lock:            
            self.cache[request_id]=item
            self.cache_status[request_id]=int(time.time()*1000)
    
    async def mark_done(self, request_id):
        if len(self.cache_status) > 30:
            now = int(time.time()*1000)
            with self.lock:
                for k in list(self.cache_status.keys()):
                    if now - self.cache_status[k] > 10*60*60*1000:
                        del self.cache_status[k]
                        del self.cache[k] 
        with self.lock:            
            self.cache_status[request_id] = 0

    async def get_item(self, request_id):                
        with self.lock:
            v = self.cache.get(request_id, None)     
            if request_id in self.cache_status and self.cache_status[request_id] == 0:
                del self.cache[request_id]
                del self.cache_status[request_id]
            return v
        
def get_type_name(t):
    name = str(t)
    if "list" in name or "dict" in name:
        return name
    else:
        return t.__name__
    
def is_annotated_type(hint):
    if hasattr(typing, '_AnnotatedAlias'):  # Python 3.9 and later
        return isinstance(hint, typing._AnnotatedAlias)
    elif hasattr(typing, '_SpecialForm'):  # Python versions before 3.9
        # Check if it's a _SpecialForm and its name is 'Annotated'
        return isinstance(hint, typing._SpecialForm) and hint.__name__ == 'Annotated'
    else:
        return False    
    
def serialize_function_to_json(func):
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    function_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "returns": type_hints.get('return', 'void').__name__
    }

    for name, parameter in signature.parameters.items():
        param_type = get_type_name(type_hints.get(name, type(None)))
        param_annotated= func.__annotations__.get(name, '')

        function_info["parameters"]["properties"][name]  = {}
        properties = function_info["parameters"]["properties"][name] 

        
        if is_annotated_type(param_annotated):
            _, *metadata = get_args(param_annotated)
        else:
            metadata = []  
   
        param_desc = ""
        for meta in metadata:
            if isinstance(meta, str):
                param_desc = meta 
            if isinstance(meta, Dict):
                param_desc = meta.get("description", "")
                if "enum" in meta:
                    properties["enum"] = meta["enum"]

        properties["type"] = param_type
        properties["description"] = param_desc
        
        if parameter.default is not inspect.Parameter.empty:
            properties["default"] = parameter.default                            

    return json.dumps(function_info,ensure_ascii=False, indent=2)


class FunctionCall(pydantic.BaseModel):
    '''
    函数名称和函数参数列表
    '''        
    name: str = pydantic.Field(description="函数名")
    arguments: Dict[str,Any] = pydantic.Field(description="函数参数")

class FunctionCallWrapper(pydantic.BaseModel):    
    function: FunctionCall = pydantic.Field(description="函数调用")

class FunctionCallList(pydantic.BaseModel):
    '''
    函数调用列表    
    '''
    tool_calls: List[FunctionCallWrapper] = pydantic.Field(description="函数调用列表")
    id: str = pydantic.Field(description="工具调用的唯一标识符,无需生成")
    type: str = pydantic.Field("function",description="工具调用的类型，固定为 function，无需生成")

FUNCTION_CALLING_SCHEMA = FunctionCallList.schema_json(ensure_ascii=False, indent=2) 


def function_calling_format(prompt:str,tools:List[Callable],tool_choice:Callable)->str:
    tool_serializes = []
    for v in tools:
        tool_serializes.append(serialize_function_to_json(v))

    force_prompt = ""
    if tool_choice is not None:
        tool_choice_ser = serialize_function_to_json(tool_choice)
        force_prompt = f''''
你必须使用如下的工具来解决用户的问题：        
```json
{tool_choice_ser}
```
'''  
   
    if tool_choice is None and len(tools) == 0:
        return prompt                   

    tools_str = "\n".join(tool_serializes)
    msg = f'''
You are a helpful assistant with access to the following functions:

```json
{tools_str}
```

{force_prompt}

当用户的问题可以使用上面的一个或者多个工具解决时,你需要生成json格式进行回复。

下面是使用 OpenAPI 3.1. 规范描述了你需如何进行json格式的生成。

```json
{FUNCTION_CALLING_SCHEMA}
```

现在用户的问题是：{prompt}

请根据描述生成 json 并发送给我。
注意：如果你无法使用上述函数解决用户的问题，请如实告诉我你没有办法回答。
''' 
    return msg  


def response_class_format(prompt:str,cls:pydantic.BaseModel)->str:
    
    msg = f'''当你回答用户问题的时候，你的输出需要是 Json 格式。
下面是使用 OpenAPI 3.1. 规范描述了你需如何进行json格式的生成。

```json
{cls.schema_json(ensure_ascii=False)}
```

现在用户的问题是：{prompt}

请根据描述生成 json 并发送给我。
''' 
    return msg 


def response_class_format_after_chat(cls:pydantic.BaseModel)->str:
    
    msg = f'''请你把刚才的回复使用 json 进行格式化。
下面是使用 OpenAPI 3.1. 规范描述了你需如何进行json格式的生成。

```json
{cls.schema_json(ensure_ascii=False)}
```

请根据描述生成 json 并发送给我。
''' 
    return msg 

  



