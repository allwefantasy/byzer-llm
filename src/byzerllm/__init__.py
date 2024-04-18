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

class _PromptWraper():        
    def __init__(self,func,llm,render,check_result,options,*args,**kwargs) -> None:
        self.func = func
        self.llm = llm
        self.render = render
        self.check_result = check_result
        self.args = args
        self.kwargs = kwargs     
        self._options = options
    
    def options(self,options:Dict[str,Any]):
        self._options = {**self._options,**options}
        return self   
    
    def with_llm(self,llm):
        self.llm = llm
        return self
    
    def prompt(self):  
        func = self.func        
        render = self.render
        args = self.args
        kwargs = self.kwargs           
                                 
        signature = inspect.signature(func)                            
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        input_dict = {}
        for param in signature.parameters:
            input_dict.update({ param: arguments.arguments[param] })          
                    
        new_input_dic = func(**input_dict)                
        if new_input_dic and not isinstance(new_input_dic,dict):
            raise TypeError(f"Return value of {func.__name__} should be a dict")                
        if new_input_dic:
            input_dict = {**input_dict,**new_input_dic}
        
        if render == "jinja2" or render == "jinja":            
            return format_prompt_jinja2(func,**input_dict)
        
        return format_prompt(func,**input_dict) 
        
    def run(self):        
        func = self.func
        llm = self.llm
        render = self.render
        check_result = self.check_result
        args = self.args
        kwargs = self.kwargs
        
        signature = inspect.signature(func)                       
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        input_dict = {}
        for param in signature.parameters:
            input_dict.update({ param: arguments.arguments[param] })         

        is_lambda = inspect.isfunction(llm) and llm.__name__ == '<lambda>'
        if is_lambda:    
            if "self" in input_dict:
                instance = input_dict.pop("self")                                                                            
                return llm(instance).prompt(render=render,check_result=check_result,options=self._options)(func)(instance,**input_dict)
            
        if isinstance(llm,ByzerLLM):
            if "self" in input_dict:
                instance = input_dict.pop("self")                                                                 
                return llm.prompt(render=render,check_result=check_result,options=self._options)(func)(instance,**input_dict)
            else:    
                return llm.prompt(render=render,check_result=check_result,options=self._options)(func)(**input_dict)
        
        if isinstance(llm,str):
            _llm = ByzerLLM()
            _llm.setup_default_model_name(llm)
            _llm.setup_template(llm,"auto")
            
            if "self" in input_dict:
                instance = input_dict.pop("self")                                                                 
                return _llm.prompt(render=render,check_result=check_result,options=self._options)(func)(instance,**input_dict)
            else:    
                return _llm.prompt(render=render,check_result=check_result,options=self._options)(func)(**input_dict)    

        
        raise ValueError("llm should be a lambda function or ByzerLLM instance or a string of model name")   
    
def prompt_lazy(llm=None,render:str="jinja2",check_result:bool=False,options:Dict[str,Any]={}):    
    def _impl(func):                                   
        @functools.wraps(func)
        def wrapper(*args, **kwargs):            
            pw = _PromptWraper(func,llm,render,check_result,options,*args,**kwargs)
            return pw

        return wrapper      
    return _impl

class _PrompRunner:
    def __init__(self,func,instance,llm,render:str,check_result:bool,options:Dict[str,Any]) -> None:
        self.func = func   
        self.instance = instance
        self.llm = llm
        self.render = render
        self.check_result = check_result
        self._options = options

    def __call__(self, *args,**kwargs) -> Any:    
        if self.llm:
            return self.run(*args, **kwargs)
        return self.prompt(*args, **kwargs)
    
    def options(self,options:Dict[str,Any]):
        self._options = {**self._options,**options}
        return self
    
    def prompt(self,*args, **kwargs):
        signature = inspect.signature(self.func)                
        if self.instance:                                   
            arguments = signature.bind(self.instance,*args, **kwargs) 
        else:
            arguments = signature.bind(*args, **kwargs)

        arguments.apply_defaults()
        input_dict = {}
        for param in signature.parameters:
            input_dict.update({ param: arguments.arguments[param] })          
                    
        new_input_dic = self.func(**input_dict)                
        if new_input_dic and not isinstance(new_input_dic,dict):
            raise TypeError(f"Return value of {self.func.__name__} should be a dict")                
        if new_input_dic:
            input_dict = {**input_dict,**new_input_dic}

        
        if "self" in input_dict:
            input_dict.pop("self")
        
        if self.render == "jinja2" or self.render == "jinja":            
            return format_prompt_jinja2(self.func,**input_dict)
        
        return format_prompt(self.func,**input_dict)
    
    def with_llm(self,llm):
        self.llm = llm
        return self
     
    def run(self,*args,**kwargs):
        func = self.func
        llm = self.llm
        render = self.render
        check_result = self.check_result        
        
        signature = inspect.signature(func)                       
        if self.instance:                                   
            arguments = signature.bind(self.instance,*args, **kwargs) 
        else:
            arguments = signature.bind(*args, **kwargs)

        arguments.apply_defaults()
        input_dict = {}
        for param in signature.parameters:
            input_dict.update({ param: arguments.arguments[param] })         

        is_lambda = inspect.isfunction(llm) and llm.__name__ == '<lambda>'
        if is_lambda:                          
            return llm(self.instance).prompt(render=render,check_result=check_result,options=self._options)(func)(**input_dict)
            
        if isinstance(llm,ByzerLLM):
            return llm.prompt(render=render,check_result=check_result,options=self._options)(func)(**input_dict)
        
        if isinstance(llm,str):
            _llm = ByzerLLM()
            _llm.setup_default_model_name(llm)
            _llm.setup_template(llm,"auto")         
            return _llm.prompt(render=render,check_result=check_result,options=self._options)(func)(**input_dict)  
                
        else:
            raise ValueError("llm should be a lambda function or ByzerLLM instance or a string of model name")  
    

class _DescriptorPrompt:    
    def __init__(self, func, wrapper,llm,render:str,check_result:bool,options:Dict[str,Any]):
        self.func = func
        self.wrapper = wrapper
        self.llm = llm
        self.render = render
        self.check_result = check_result
        self._options = options
        self.prompt_runner = _PrompRunner(self.wrapper,None,self.llm,self.render,self.check_result,options=self._options)

    def __get__(self, instance, owner):        
        if instance is None:
            return self
        else:            
            return _PrompRunner(self.wrapper,instance,self.llm,self.render,self.check_result,options=self._options)

    def __call__(self, *args, **kwargs):
        return self.prompt_runner(*args, **kwargs)

    def prompt(self,*args, **kwargs):
        return self.prompt_runner.prompt(*args, **kwargs)

    def run(self,*args, **kwargs):
        return self.prompt_runner.run(*args, **kwargs)

    def with_llm(self,llm):
        self.llm = llm
        self.prompt_runner.with_llm(llm)
        return self  

    def options(self,options:Dict[str,Any]):
        self._options = {**self._options,**options}        
        self.prompt_runner.options(options)
        return self  

class prompt:
    def __init__(self, llm=None,render:str="jinja2",check_result:bool=False,options:Dict[str,Any]={}):
        self.llm = llm
        self.render = render
        self.check_result = check_result
        self.options = options

    def __call__(self, func):        
        # if 'self' in func.__code__.co_varnames:            
        #     wrapper = func            
        #     return self._make_wrapper(func, wrapper)
        # return _PrompRunner(func,None,self.llm,self.render,self.check_result)
        wrapper = func            
        return self._make_wrapper(func, wrapper)

    def _make_wrapper(self, func, wrapper):            
        return _DescriptorPrompt(func, wrapper,self.llm,self.render,self.check_result,options=self.options)



from byzerllm.utils.client import ByzerLLM
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.connect_ray import connect_cluster
from byzerllm.apps.agent.registry import reply as agent_reply
__all__ = ["ByzerLLM","ByzerRetrieval","connect_cluster","prompt","agent_reply"]

       
    