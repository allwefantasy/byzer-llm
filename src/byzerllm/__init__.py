from typing import Any,List,Dict
from ray.util.client.common import ClientObjectRef
from pyjava.api.mlsql import RayContext
from pyjava.storage import streaming_tar
import os
import inspect
import functools

from typing import Dict,Generator,Optional
from dataclasses import dataclass
from byzerllm.utils import (print_flush,format_prompt,format_prompt_jinja2)
from .store import transfer_from_ob

@dataclass
class BlockRow:
    start: int
    offset: int
    value: bytes

def restore_model(conf: Dict[str, str],target_dir:str):
    model_servers = RayContext.parse_servers(conf["modelServers"])
    model_binary = RayContext.collect_from(model_servers)
    streaming_tar.save_rows_as_file(model_binary,target_dir)

def load_model(target_dir:str)-> Generator[BlockRow,None,None]:
    model_binary = streaming_tar.build_rows_from_file(target_dir)
    return model_binary

def consume_model(conf: Dict[str, str]):
    # consume the model server to prevent socket server leak.
    # hoverer,  model server may have be consumed by other worker
    # so just try to consume it
    try:
        model_servers = RayContext.parse_servers(conf["modelServers"])
        for item in RayContext.collect_from(model_servers):
            pass
    except Exception as e:
        pass   

def common_init_model(model_refs: List[ClientObjectRef], 
                      conf: Dict[str, str],model_dir:str,is_load_from_local:bool):
    
    udf_name  = conf["UDF_CLIENT"] if "UDF_CLIENT" in conf else "UNKNOW MODEL"

    if not is_load_from_local:      
      if "standalone" in conf and conf["standalone"]=="true":
          print_flush(f"MODEL[{udf_name}] Standalone mode: restore model to {model_dir} directly from model server")
          restore_model(conf,model_dir)
      else:
          print_flush(f"MODEL[{udf_name}] Normal mode: restore model from ray object store to {model_dir}")
          if not os.path.exists(model_dir):
            transfer_from_ob(udf_name,model_refs,model_dir)
    else:
      print_flush(f"MODEL[{udf_name}]  Local mode: Load model from local path ({model_dir}), consume the model server to prevent socket server leak.")
      consume_model(conf)   

def parse_params(params:Dict[str,str],prefix:str):
    import json
    new_params = {}
    for k,v in params.items():
        if k.startswith(f"{prefix}."):
            # sft.float.num_train_epochs
            tpe = k.split(".")[1]
            new_k = k.split(".")[2]
            new_v = v
            if tpe == "float":
              new_v = float(v)
            elif tpe == "int":
                new_v = int(v)
            elif tpe == "bool":
                new_v = v == "true"
            elif tpe == "str":
                new_v = v
            elif tpe == "list":
                new_v = json.loads(v)
            elif tpe == "dict":
                new_v = json.loads(v)            
            new_params[new_k] = new_v
    return new_params 

import inspect


def check_param_exists(func,name):
    return name in inspect.signature(func).parameters


# add a log funcition to log the string to a specified file
def log_to_file(msg:str,file_path:str):
    with open(file_path,"a") as f:
        f.write(msg)
        f.write("\n")


def prompt(llm=None,render:str="simple",print_prompt:bool=False):    
    '''
    decorator to add prompt function to a method
    render: simple,jinja/jinja2
    llm: lambda function to get the ByzerLLM instance from the method instance
    '''
    def _impl(func):                       
        @functools.wraps(func)
        def wrapper(*args, **kwargs):                      
            signature = inspect.signature(func)
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()
            input_dict = {}
            for param in signature.parameters:
                input_dict.update({ param: arguments.arguments[param] })
            if "self" in input_dict:
                instance = input_dict.pop("self") 
                is_lambda = inspect.isfunction(llm) and llm.__name__ == '<lambda>'                
                if is_lambda:                                        
                    return llm(instance).prompt(render=render,print_prompt=print_prompt)(func)(instance,**input_dict)             
            
            if render == "jinja2" or render == "jinja":                  
                return format_prompt_jinja2(func,**input_dict)
            
            return format_prompt(func,**input_dict) 

        return wrapper      
    return _impl


from byzerllm.utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.connect_ray import connect_cluster
from byzerllm.apps.agent.registry import reply as agent_reply
__all__ = ["ByzerLLM","ByzerRetrieval","connect_cluster","prompt","agent_reply"]

       
    