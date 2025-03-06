from ..conversable_agent import ConversableAgent
from ..agent import Agent
from .. import get_agent_name,run_agent_func,ChatResponse
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from typing import Callable, Dict, Optional, Union
import time
import sys
import io
import traceback
import json
import os
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.client import ByzerLLM,code_utils,message_utils



class CodeSandbox:
    def __init__(self,file_path:str,file_ref) -> None:        
        self.file_ref = file_ref
        self.file_path = file_path
        self.session_variables = {}
        if self.file_ref:
            if isinstance(self.file_ref,ClientObjectRef):
                content = ray.get(self.file_ref)
            else:
                content = self.file_ref   

            # check parent directory of self.file_path exists
            parent_dir = os.path.dirname(self.file_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)     
                     
            with open(self.file_path, "wb") as f:
                f.write(content)
                
    def set_value(self,name:str,value:str): 
        self.session_variables[name] = value
        return self

    def get_value(self,name:str):
        if name not in self.session_variables:
            return None
        return self.session_variables[name]

    def get_file_path(self):
        return self.file_path        

    def execute_code(self,code)->Tuple[int, str, str]:
        return code_utils.execute_code(
                code = code,
                timeout=30*60,
                filename=None,
                work_dir=None,
                use_docker=False,
                lang="python"        
                ) 
    
    def exec_capture_output(self,code: str,target_names:Dict[str,Any]={}) -> Tuple[int,str,Any]:
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
    
class PythonSandboxAgent(ConversableAgent):        

    def __init__(
        self,
        name: str,
        llm: ByzerLLM,
        retrieval: ByzerRetrieval,   
        chat_name:str,
        owner:str,     
        system_message: Optional[str],        
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[Dict, bool]] = {},
        **kwargs,
    ):
        super().__init__(
            name,
            llm,retrieval,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            code_execution_config=code_execution_config,            
            **kwargs,
        )
        self.sandboxes = {}
        self.lasted_updated = {}
        
        ## Restore the reply function list
        self._reply_func_list = []        

        ## Register the reply functions                
        self.register_reply([Agent, ClientActorHandle,str], PythonSandboxAgent.generate_execute_code_reply) 
        self.register_reply([Agent, ClientActorHandle,str], ConversableAgent.check_termination_and_human_reply)             
        

    def generate_execute_code_reply(
        self,
        raw_message: Optional[Union[Dict,str,ChatResponse]] = None,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Union[ClientActorHandle,Agent,str]] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None,ChatResponse]]:
                
        code_execution_config = config if config is not None else self._code_execution_config
        
        if code_execution_config is False:            
            return False, None
        
        if messages is None:
            messages = self._messages[get_agent_name(sender)]
        
        last_n_messages = code_execution_config.pop("last_n_messages", 1)                

        for i in range(min(len(messages), last_n_messages)):
            message = messages[-(i + 1)]
            if not message["content"]:
                continue            
            
            code_blocks = code_utils.extract_code(message["content"])
            if len(code_blocks) == 1 and code_blocks[0][0] == "unknown":
                continue

            # found code blocks, execute code and push "last_n_messages" back
            #  combine all code blocks into one code block
            codes = [code_block[1] for code_block in code_blocks if code_block[0] == "python"]
            code_str = "\n".join(codes)
            
            file_path = None
            file_ref = None

            if "metadata" not in message:
                message["metadata"] = {}
            
            if "file_path" in message["metadata"]:
                file_path = message["metadata"]["file_path"]
                file_ref = message["metadata"]["file_ref"]                
            
            target_names = {}
            if "target_names" in message["metadata"]:
                target_names = message["metadata"]["target_names"]

            sandbox = self.get_or_create_sandbox(get_agent_name(sender)+"_sandbox",file_path,file_ref,0,0)            
            exitcode, output,response = ray.get(sandbox.exec_capture_output.remote(code_str,target_names))
            code_execution_config["last_n_messages"] = last_n_messages
            exitcode2str = "execution succeeded" if exitcode == 0 else "execution failed"
            
            return True, {
                "content":f"exitcode: {exitcode} ({exitcode2str})\nCode output: {output}",
                 "metadata":{
                     "execute_result":ChatResponse(status=exitcode,
                                      output=f"exitcode: {exitcode} ({exitcode2str})\nCode output: {output}",
                                      code=code_str,
                                      prompt=message,
                                      variables=response,                                      
                                      ),
                      "error_count": message_utils.get_error_count(message),
                 }
            }

        print("No code block found in the last {} messages.".format(last_n_messages),flush=True)
        code_execution_config["last_n_messages"] = last_n_messages

        return True, None            

    def check_sandbox_timeout(self,timeout:int=60*60): 
        remove_names = []
        for name in self.lasted_updated:
            if time.time() - self.lasted_updated[name] > timeout:
                remove_names.append(name)
        for name in remove_names:
            del self.sandboxes[name]
            del self.lasted_updated[name]        

    def check_sandbox_exists(self,name:str)->bool:
        return name in self.sandboxes

    def get_sandbox(self,name:str):                
        self.check_sandbox_timeout()        
        return self.sandboxes[name]
    
    def force_clear(self):
        self.sandboxes = {}
        self.lasted_updated = {}

    def get_or_create_sandbox(self,name:str,
                              file_path:str,file_ref:str,
                              num_gpus:int,num_cpus:int):
        self.lasted_updated[name] = time.time()
        self.check_sandbox_timeout()
        if name in self.sandboxes:            
            return self.sandboxes[name]
        
        try :
            sandbox = ray.get_actor(name)
            return sandbox
        except ValueError:
            pass
        
        sandbox = ray.remote(CodeSandbox).options(
                name=name,                                             
                num_cpus=num_cpus,
                num_gpus=num_gpus
            ).remote(file_path,file_ref)
        self.sandboxes[name] = sandbox
        return sandbox    