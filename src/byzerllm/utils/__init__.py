import uuid
from pathlib import Path
from functools import wraps
import time
import json
import hashlib
from typing import TYPE_CHECKING,TypeVar,Dict, List, Optional, Union,Any,Tuple,get_type_hints,Annotated,get_args,Callable
import typing
import inspect
import pydantic
import sys
import traceback
import io
from enum import Enum
from byzerllm.utils.types import BlockVLLMStreamServer,StreamOutputs,SingleOutput,SingleOutputMeta,BlockBinaryStreamServer

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

def tokenize_string(tokenizer, key: str) -> Union[int, List[int]]:
    """Tokenize a string using a tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        key (str): String to tokenize.
    """
    token_ids = tokenizer.encode(key, add_special_tokens=False)
    return token_ids

def tokenize_stopping_sequences_where_needed(
    tokenizer,
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
    if isinstance(func, str):
        return func
    
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    # return_type = type_hints.get('return', 'void')
    # if return_type is None:
    #     return_type_str = 'void'
    # else:
    #     return_type_str = return_type.__name__

    function_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {
            "type": "object",
            "properties": {}
        },
        # "returns": return_type_str
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


def exec_capture_output(code: str,target_names:Dict[str,Any]={}) -> Tuple[int,str,Any]:
    buffer = io.StringIO()
    sys.stdout = buffer
    sys.stderr = buffer

    try:
        variables = {}
        exec(code,variables)
        response = {}
        for name,v in target_names.items():
            if name in variables:
                response[name] = variables[name]
    except Exception:
        return 1,traceback.format_exc(),{}

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return 0,buffer.getvalue(),response

def function_impl_format(prompt:str,func:Optional[Union[Callable,str]],
                             cls:Union[pydantic.BaseModel,str])->str:
    
    tool_choice_ser = serialize_function_to_json(func)    
    _cls = ""
    if isinstance(cls, str):
        _cls = cls
    else:
        _cls = cls.schema_json(ensure_ascii=False)

    example = '''{
  "name": "caculate_current_time",
  "description": "\n    计算当前时间\n    ",
  "parameters": {
    "type": "object",
    "properties": {}
  }
}'''    
    example_output_format='''
{"title": "CurrentTime", "description": "当前时间    ", "type": "object", "properties": {"time": {"title": "Time", "description": "开始时间.时间格式为 yyyy-MM-dd", "type": "string"}}, "required": ["time"]}
'''
    example_output = '''
from datetime import datetime

def caculate_current_time():
    # 获取当前日期和时间
    now = datetime.now()
    
    # 将日期和时间格式化为"yyyy-MM-dd"的形式
    time_str = now.strftime("%Y-%m-%d")
    
    return {"time": time_str}
'''
    
    msg = f''''你非常擅长 Python 语言。根据用户提供的一些信息以及问题，对提供了没有实现空函数函数进行实现。

示例：
你需要实现的函数的签名如下：

```json
{example}
```

生成的函数的返回值必须是 Json 格式，并且满足如下 OpenAPI 3.1. 规范：

```json
{example_output_format}
```

最后，你生成的函数的代码如下：

```python
{example_output}
```

现在，你需要实现函数的签名如下：

```json
{tool_choice_ser}
```

同时，你生成的函数的返回值必须是 Json 格式，并且满足如下 OpenAPI 3.1. 规范：

```json
{_cls}
```

用户的问题是：{prompt}

在满足上述提及的约束的情况下，请你实现这个函数。
注意：
1. 任何情况下都不要拆分成多段代码输出，请一次性生成完整的代码片段，确保代码的完整性
2. 回复的内容只有一个代码块，且代码块的语言为 Python
3. 不要演示如何调用你生成的函数的代码
'''
    return msg   



def function_calling_format(prompt:str,tools:List[Union[Callable,str]],tool_choice:Optional[Union[Callable,str]])->str:
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
    
    function_example = '''
{
  "name": "compute_date_range",
  "description": "\n    计算日期范围\n    ",
  "parameters": {
    "type": "object",
    "properties": {
      "count": {
        "type": "int",
        "description": "时间跨度，数值类型,如果用户说的是几天，几月啥的，比较模糊，务必使用默认值",
        "default": 3
      },
      "unit": {
        "enum": [
          "day",
          "week",
          "month",
          "year"
        ],
        "type": "str",
        "description": "",
        "default": "day"
      }
    }
  }
}
'''
    output_example = '''
{
  "id": "unique_id_1",
  "type": "function",
  "tool_calls": [
    {
      "function": {
        "name": "compute_date_range",
        "arguments": {
          "count": 3,
          "unit": "day"
        }
      }
    }
  ]
}
'''

    msg = f'''You are a helpful assistant with access to the following functions:

```json
{tools_str}
```

当用户的问题可以使用上面的一个或者多个函数解决时,你需要通过符合 OpenAPI 3.1 规范的 Json 格式告诉我你需要调用哪些函数。

下面Json文本描述了你需要返回的格式,它符合 OpenAPI 3.1 规范:

```json
{FUNCTION_CALLING_SCHEMA}
```

示例：

当你选择下面的函数时：

```
{function_example}
```

你应该使用如下的 Json 格式告诉我你需要调用这个函数：

```json
{output_example}
```

{force_prompt}

现在用户的问题是：{prompt}

请选择合适的一个或者多个函数按要求的 Json 格式返回给我。

注意：
1. 如果你无法使用上述函数解决用户的问题，请如实告诉我你没有办法回答。
''' 
    return msg  


def response_class_format(prompt:str,cls:Union[pydantic.BaseModel,str])->str:

    _cls = ""
    if isinstance(cls, str):
        _cls = cls
    else:
        _cls = cls.schema_json(ensure_ascii=False)    
        
    example='''
{"title": "Item", "description": "时间抽取的返回结果", "type": "object", "properties": {"time": {"title": "Time", "description": "时间信息,比如内容里会提到天， 月份，年等相关词汇", "type": "string"}, "other": {"title": "Other", "description": "除了时间以外的其他部分", "type": "string"}}, "required": ["time", "other"]}
'''
    example_output = '''{
  "time": "最近三个月",
  "other": "奔驰的销量趋势如何"
}'''
    msg = f'''当你回答用户问题的时候，你需要使用 Json 格式进行回复。

示例：

当你被要求按如下格式输出时,它符合 OpenAPI 3.1 规范：

```json
{example}
```
你的输出应该是这样的：

```json
{example_output}
```

现在用户的问题是：{prompt}

下面Json文本描述了你需要返回的格式,它符合 OpenAPI 3.1 规范:

```json
{_cls}
```

请根据自己生成的内容并以 Json 格式回复我。
''' 
    return msg

def response_class_format_after_chat(cls:Union[pydantic.BaseModel,str])->str:
 
    _cls = ""
    if isinstance(cls, str):
        _cls = cls
    else:
        _cls = cls.schema_json(ensure_ascii=False)
    example='''
{"title": "Item", "description": "时间抽取的返回结果", "type": "object", "properties": {"time": {"title": "Time", "description": "时间信息,比如内容里会提到天， 月份，年等相关词汇", "type": "string"}, "other": {"title": "Other", "description": "除了时间以外的其他部分", "type": "string"}}, "required": ["time", "other"]}
'''
    example_output = '''{
  "car": {
    "name": "奔驰"
  },
  "metric": {
    "name": "销量趋势"
  }'''    
    msg = f'''你需要以 Json 格式重新组织内容回复我。

示例：

当你被要求按如下格式输出时,它符合 OpenAPI 3.1 规范：

```json
{example}
```
你的输出应该是这样的：

```json
{example_output}
```
把你刚才回答我的内容重新做组织，以 Json 格式回复我

下面Json文本描述了你需要返回的格式,它符合 OpenAPI 3.1 规范:

```json
{_cls}
```
''' 
    return msg 

class BaseAbility(Enum):
    RESPONSE_WITH_CLASS = "RESPONSE_WITH_CLASS"
    FUNCTION_CALLING = "FUNCTION_CALLING"
    FUNCTION_IMPL = "FUNCTION_IMPL"
    OTHERS = "OTHERS"

def base_ability_format(prompt:Optional[str]=None,base_abilities: List[BaseAbility]=[BaseAbility.FUNCTION_CALLING,
                                                                                     BaseAbility.FUNCTION_IMPL,
                                                                                     BaseAbility.RESPONSE_WITH_CLASS                                                                                     
                                                                                     ])->str:
    base_abilities.extend([BaseAbility.OTHERS])

    RESPONSE_WITH_CLASS_example_0='''{"title": "Item", "description": "时间抽取的返回结果", "type": "object", "properties": {"time": {"title": "Time", "description": "时间信息,比如内容里会提到天， 月份，年等相关词汇", "type": "string"}, "other": {"title": "Other", "description": "除了时间以外的其他部分", "type": "string"}}, "required": ["time", "other"]}'''
    RESPONSE_WITH_CLASS_example_output_0 = '''{
    "time": "最近三个月",
    "other": "奔驰的销量趋势如何"
    }'''

    RESPONSE_WITH_CLASS_example='''{"title": "Info", "type": "object", "properties": {"car": {"title": "Car", "description": "车的信息", "allOf": [{"$ref": "#/definitions/Car"}]}, "metric": {"title": "Metric", "description": "计算的指标信息", "allOf": [{"$ref": "#/definitions/Metric"}]}}, "required": ["car", "metric"], "definitions": {"Car": {"title": "Car", "type": "object", "properties": {"name": {"title": "Name", "description": "品牌名称", "type": "string"}}, "required": ["name"]}, "Metric": {"title": "Metric", "type": "object", "properties": {"name": {"title": "Name", "description": "指标名称", "type": "string"}}, "required": ["name"]}}}'''
    RESPONSE_WITH_CLASS_example_output = '''{
  "car": {
    "name": "奔驰"
  },
  "metric": {
    "name": "销量趋势"
  }
}'''
    RESPONSE_WITH_CLASS_example='''{"title": "Info", "type": "object", "properties": {"car": {"title": "Car", "description": "车的信息", "allOf": [{"$ref": "#/definitions/Car"}]}, "metric": {"title": "Metric", "description": "计算的指标信息", "allOf": [{"$ref": "#/definitions/Metric"}]}}, "required": ["car", "metric"], "definitions": {"Car": {"title": "Car", "type": "object", "properties": {"name": {"title": "Name", "description": "品牌名称", "type": "string"}}, "required": ["name"]}, "Metric": {"title": "Metric", "type": "object", "properties": {"name": {"title": "Name", "description": "指标名称", "type": "string"}}, "required": ["name"]}}}'''
    RESPONSE_WITH_CLASS_example_output = '''{
  "car": {
    "name": "奔驰"
  },
  "metric": {
    "name": "销量趋势"
  }
}'''

    FUNCTION_CALLING_example = '''
{
  "name": "compute_date_range",
  "description": "\n    计算日期范围\n    ",
  "parameters": {
    "type": "object",
    "properties": {
      "count": {
        "type": "int",
        "description": "时间跨度，数值类型,如果用户说的是几天，几月啥的，比较模糊，务必使用默认值",
        "default": 3
      },
      "unit": {
        "enum": [
          "day",
          "week",
          "month",
          "year"
        ],
        "type": "str",
        "description": "",
        "default": "day"
      }
    }
  }
}
'''
    FUNCTION_CALLING_example_output = '''
{
  "id": "unique_id_1",
  "type": "function",
  "tool_calls": [
    {
      "function": {
        "name": "compute_date_range",
        "arguments": {
          "count": 3,
          "unit": "day"
        }
      }
    }
  ]
}
'''

    FUNCTION_IMPL_example = '''{
  "name": "caculate_current_time",
  "description": "\n    计算当前时间\n    ",
  "parameters": {
    "type": "object",
    "properties": {}
  }
}'''    
    FUNCTION_IMPL_example_output_schema='''
{"title": "CurrentTime", "description": "当前时间    ", "type": "object", "properties": {"time": {"title": "Time", "description": "开始时间.时间格式为 yyyy-MM-dd", "type": "string"}}, "required": ["time"]}
'''
    FUNCTION_IMPL_example_output = '''
from datetime import datetime

def caculate_current_time():
    # 获取当前日期和时间
    now = datetime.now()
    
    # 将日期和时间格式化为"yyyy-MM-dd"的形式
    time_str = now.strftime("%Y-%m-%d")
    
    return {"time": time_str}
'''
    m_response_class_str = "" if BaseAbility.RESPONSE_WITH_CLASS not in base_abilities else f'''
===================RESPONSE_WITH_CLASS===================

下面是一个根据用户的问题，并且结合 JSON Schema 生成对应的 JSON 数据的例子：

输入：

最近三个月奔驰的销量趋势如何？

JSON Schema：

```json
{RESPONSE_WITH_CLASS_example_0}
```

输出：

```json
{RESPONSE_WITH_CLASS_example_output_0}
```

下面生成的 Json 数据有有嵌套结构的例子：

输入：

最近三个月奔驰的销量趋势如何？

JSON Schema：

```json
{RESPONSE_WITH_CLASS_example}
```

输出：

```json
{RESPONSE_WITH_CLASS_example_output}
```

当用户提到 RESPONSE_WITH_CLASS 时，请回顾该能力。
'''
    m_function_calling_str = "" if BaseAbility.FUNCTION_CALLING not in base_abilities else f'''
===================FUNCTION_CALLING===================

用户会提供一个函数列表给你,你需要根据用户的问题，选择一个或者多个函数返回给用户。如果你无法使用上述函数解决用户的问题，请如实告诉我你没有办法回答。
下面假设你已经选择了一个函数作为输入，并且结合 JSON Schema 生成对应的 JSON 数据的例子：

输入：

```json
{FUNCTION_CALLING_example}
```

JSON Schema：

```json
{FUNCTION_CALLING_SCHEMA}
```

输出：

```json
{FUNCTION_CALLING_example_output}
```

当用户提到 FUNCTION_CALLING 时，请回顾该能力。
''' 
    m_function_impl_str = "" if BaseAbility.FUNCTION_IMPL not in base_abilities else f'''
===================FUNCTION_IMPL===================

你非常擅长 Python 语言。根据用户提供的一些信息以及问题，对用户提供的没有实现空函数函数进行实现。
下面假设用户提供了一个需要实现的函数的签名，你需要结合用户的问题，函数的签名，以及函数文档，生成对应的 Python 代码，函数的返回值
必须是 Json 格式，并且需要符合对应的 JSON Schema 规范。

下面提供了一个示例：

输入：

```json
{FUNCTION_IMPL_example}
```

JSON Schema：

```json
{FUNCTION_IMPL_example_output_schema}
```

输出：

```python
{FUNCTION_IMPL_example_output}
```

注意：
1. 任何情况下都不要拆分成多段代码输出，请一次性生成完整的代码片段，确保代码的完整性
2. 回复的内容只有一个代码块，且代码块的语言为 Python
3. 不要展示如何调用你生成的函数的代码
4. 不要展示你函数执行的结果

当用户提到 FUNCTION_IMPL 时，请回顾该能力。
'''
    msg = f'''下面是你具备的基础能力，当你回答用户问题的时候，随时回顾这些能力。

JSON 格式是一种轻量级的数据交换格式，JSON Schema 是基于 JSON 的一个描述 JSON 数据结构的元数据，可以用来描述 JSON 数据的结构和内容，以及定义 JSON 数据的合法值范围。
OpenAPI Specification (OAS) 使用 JSON Schema 来描述 Json 数据的结构和内容，你需要遵循 OpenAPI 3.1.0 版本的规范。

{m_response_class_str}

{m_function_calling_str}

{m_function_impl_str}

===================OTHERS===================
'''
    
    return msg


def sys_response_class_format(prompt:str,cls:Union[pydantic.BaseModel,str])->str:
    
    _cls = ""
    if isinstance(cls, str):
        _cls = cls
    else:
        _cls = cls.schema_json(ensure_ascii=False)

    msg = f'''
请使用 RESPONSE_WITH_CLASS 相关的能力，解决用户的问题。

输入：

{prompt}

JSON Schema：

```json
{_cls}
```

输出：
'''
    return msg

def sys_function_calling_format(prompt:str,tools:List[Union[Callable,str]],tool_choice:Optional[Union[Callable,str]])->str:
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
请使用 FUNCTION_CALLING 相关的能力，解决用户的问题。

你有如下的函数可以使用：

```json
{tools_str}
```
{force_prompt}

输入：

{prompt}

JSON Schema：

```json
{FUNCTION_CALLING_SCHEMA}
```

输出:
''' 
    return msg 

def sys_function_impl_format(prompt:str,func:Optional[Union[Callable,str]],
                             cls:Union[pydantic.BaseModel,str])->str:
    
    tool_choice_ser = serialize_function_to_json(func)    
    _cls = ""
    if isinstance(cls, str):
        _cls = cls
    else:
        _cls = cls.schema_json(ensure_ascii=False)

    
    msg = f''''请使用 FUNCTION_IMPL 相关的能力，解决用户的问题。
根据用户提供的一些信息以及函数签名，对函数进行实现。

用户问题： {prompt}

输入：

```json
{tool_choice_ser}
```

JSON Schema：

```json
{_cls}
```

输出:
'''
    return msg  

def format_prompt(func,**kargs): 
    from langchain import PromptTemplate
    doc = func.__doc__       
    lines = doc.splitlines()
    # get the first line to get the whitespace prefix
    first_non_empty_line = next(line for line in lines if line.strip())
    prefix_whitespace_length = len(first_non_empty_line) - len(first_non_empty_line.lstrip())    
    prompt = "\n".join([line[prefix_whitespace_length:] for line in lines])
    tpl = PromptTemplate.from_template(prompt)
    return tpl.format(**kargs)

def format_prompt_jinja2(func,**kargs):
    from jinja2 import Template
    doc = func.__doc__       
    lines = doc.splitlines()
    # get the first line to get the whitespace prefix
    first_non_empty_line = next(line for line in lines if line.strip())
    prefix_whitespace_length = len(first_non_empty_line) - len(first_non_empty_line.lstrip())    
    prompt = "\n".join([line[prefix_whitespace_length:] for line in lines])
    tpl = Template(prompt)
    return tpl.render(kargs)

def random_uuid() -> str:
    return str(uuid.uuid4().hex)


__all__ = ["BlockVLLMStreamServer","StreamOutputs","SingleOutput","SingleOutputMeta","BlockBinaryStreamServer"]

