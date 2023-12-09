from pyjava.udf import UDFMaster
from pyjava import PythonContext,RayContext
from typing import Dict,Any,List,Optional,Union,Tuple,Callable
from pyjava.udf import UDFBuilder
import ray
import sys
import traceback
import io
import os
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import json
import uuid
import dataclasses
import importlib  
from . import code_utils
from . import utils
from . import LLMHistoryItem,ByzerLLM,ByzerRetrieval,LLMRequest,ExecuteCodeResponse,Role,LLMRequestExtra
from ..retrieval import ByzerRetrieval,TableSettings,SearchQuery
from .. import prompts as PROMPTS
import logging
import time
import math
from byzerllm.utils import generate_str_md5

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

class SandboxManager:
    def __init__(self) -> None:
        self.sandboxes = {}
        self.lasted_updated = {}
    
    ## if the sandbox is not used for 1h, we will remove it
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
        
        sandbox = ray.remote(CodeSandbox).options(
                name=name,                              
                num_cpus=num_cpus,
                num_gpus=num_gpus
            ).remote(file_path,file_ref)
        self.sandboxes[name] = sandbox
        return sandbox
    

class DataAnalysisMode:
    data_analysis = "data_analysis"
    text_analysis = "text_analysis" 
    auto_analysis = "auto_analysis"      

class ByzerDataAnalysis:
    def __init__(self,llm:ByzerLLM,
                 retrieval:ByzerRetrieval=None,
                 chat_name:str=None,
                 owner:str=None,
                 file_path:str= None, 
                 use_shared_disk:bool=False,
                 retrieval_cluster:str="data_analysis",
                 retrieval_db:str="data_analysis", 
                 data_analysis_mode:DataAnalysisMode=DataAnalysisMode.data_analysis, 
                 role_mapping = {
                    "user_role":"User:",
                    "assistant_role": "Assistant:",
                    "system_msg":"You are a helpful assistant. Think it over and answer the user question correctly."
                    }, 
                 max_length:int=8024,   
                 tempraure:float=0.1,
                 max_input_length=1024*4,
                 max_output_length=1200,
                 verbose:bool=False, 
                 keep_conversation:bool=True,             
                 num_gpus=0, num_cpus=1) -> None:
        self.llm = llm
        
        self.retrieval = retrieval
        self.data_analysis_mode = data_analysis_mode
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.llm.max_input_length = self.max_input_length
        

        self.use_shared_disk = use_shared_disk
        
        self.sandbox_manager = self.get_sandbox_manager()

        self.file_path = file_path
        self.file_ref = None
        self.file_preview = None
        self.loaded_successfully=False

        self.max_length = max_length
        self.tempraure = tempraure
        self.verbose = verbose
        self.keep_conversation = keep_conversation

        self.role_mapping = role_mapping

        self.retrieval_cluster = retrieval_cluster
        self.retrieval_db = retrieval_db        

        self.owner = owner
        self.chat_name = chat_name                
        
        if self.owner is None:
            self.owner = str(uuid.uuid4())            

        self.sandbox_suffix = generate_str_md5(self.owner+"_"+self.chat_name)    
        
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        
        sandbox_name = f"CodeSandbox-{self.sandbox_suffix}"

        if not self.check_sandbox_exists(sandbox_name):             
            if self.file_path and not self.use_shared_disk  and self.data_analysis_mode == DataAnalysisMode.data_analysis:
                base_name = os.path.basename(file_path)
                name, ext = os.path.splitext(base_name)
                new_base_name = self.sandbox_suffix + ext
                dir_name = os.path.dirname(file_path)
                new_file_path = os.path.join(dir_name, new_base_name)

                logger.info(f"use_shared_disk: {self.use_shared_disk} file_path: {self.file_path} new_file_path: {new_file_path}")
                self.file_ref = ray.put(open(self.file_path,"rb").read())
                self.file_path = new_file_path

            if self.file_path and self.data_analysis_mode == DataAnalysisMode.text_analysis:
                content = open(self.file_path).read()
                self.save_text_content(title="noops",owner=self.owner,content=content,url=self.file_path)

            self.get_or_create_sandbox(sandbox_name) 
        else:
            # restore value from sandbox   
            sandbox = self.get_sandbox(sandbox_name)                       
            self.file_preview = ray.get(sandbox.get_value.remote("file_preview"))
            self.file_path = ray.get(sandbox.get_file_path.remote())
            restore_loaded_successfully = ray.get(sandbox.get_value.remote("loaded_successfully"))
            self.loaded_successfully = restore_loaded_successfully if restore_loaded_successfully else False

        if self.retrieval and not self.retrieval.check_table_exists(self.retrieval_cluster,self.retrieval_db,"text_content"):
           self.retrieval.create_table(self.retrieval_cluster,tableSettings=TableSettings(
                database=self.retrieval_db,
                table="text_content",schema='''st(
field(_id,string),
field(owner,string),
field(title,string,analyze),
field(content,string,analyze),
field(url,string),
field(raw_content,string),
field(auth_tag,string,analyze),
field(title_vector,array(float)),
field(content_vector,array(float))
)''',
                location=f"/tmp/{self.retrieval_cluster}",num_shards=1                
           ))

           self.retrieval.create_table(self.retrieval_cluster,tableSettings=TableSettings(
                database=self.retrieval_db,
                table="text_content_chunk",schema='''st(
field(_id,string),
field(doc_id,string),
field(owner,string),
field(chunk,string,analyze),
field(raw_chunk,string),
field(chunk_vector,array(float))
)''',
                location=f"/tmp/{self.retrieval_cluster}",num_shards=1                
           )) 
           if not self.retrieval.check_table_exists(self.retrieval_cluster,self.retrieval_db,"user_memory"):
                self.retrieval.create_table(self.retrieval_cluster,tableSettings=TableSettings(
                        database=self.retrieval_db,
                        table="user_memory",schema='''st(
        field(_id,string),
        field(chat_name,string),
        field(role,string),
        field(owner,string),
        field(content,string,analyze),
        field(raw_content,string),
        field(auth_tag,string,analyze),
        field(created_time,long,sort),
        field(chat_name_vector,array(float)),
        field(content_vector,array(float))
        )
        ''',
                        location="",num_shards=""                
                ))   
            

    def generate_code(self, prompt:Union[str,LLMRequest],pattern: str = code_utils.CODE_BLOCK_PATTERN, **config) -> Tuple[str, float]:
        """Generate code.

        Args:
            prompt (str): The prompt for generating code.
            pattern (Optional, str): The regular expression pattern for finding the code block.
                The default pattern is for finding a code block in a markdown file.
            config (Optional, dict): The configuration for the API call.

        Returns:
            str: The generated code.
            float: The cost of the generation.
        """                
        response = self.llm.raw_chat(None,request=LLMRequest.build(instruction=prompt,
                                                                   max_length=self.max_length,
                                                                   temperature=self.tempraure,
                                                                   role_mapping=self.role_mapping),extract_params=config)
        return code_utils.extract_code(response[0].output, pattern), -1 

    def improve_function(self,file_name, func_name, objective, **config):
        """Improve the function to achieve the objective."""        
        # read the entire file into a str
        with open(file_name, "r") as f:
            file_string = f.read()
        new_prompt = f'''Improve the function '{func_name}' to achieve the objective '{objective}'.
The current implementation of the function is as follows:
{file_string}'''
        response = self.llm.raw_chat(None, request=LLMRequest(instruction=new_prompt,**config))            
        return response[0].output, -1

    def default_check_eval_repsonse(self,response:Dict[str,Any],target_names:Dict[str,Any]={})->Tuple[bool,str]:
        missing_variables = []
        
        for name,value in target_names.items():
            if name not in response:
                missing_variables.append(f'Make sure {name} is defined in the top level scope')
            elif value is not None and response[name] != value:
                missing_variables.append(f'Make sure {name} is set to the correct value. Expected: {value}, Actual: {response[name]}') 
        if not missing_variables:        
            return True ,""        
        return False,"Here are the code problems:\n"+"\n".join(missing_variables) if missing_variables else ""

    def search_tokenize(self,s:str):
        return self.llm.apply_sql_func("select mkString(' ',parse(value)) as value",[
        {"value":s}])["value"]
    
    def emb(self,s:str, emb_model:str="emb"):
        return self.llm.emb(emb_model,LLMRequest(instruction=s))[0].output

    def save_conversation(self,owner:str,role:str,content:str):
        if not self.retrieval:
            raise Exception("retrieval is not setup")                                

        if self.chat_name is None:
            self.chat_name = content[0:10]   

        if len(content) > self.max_output_length:
            raise Exception(f"The response content length {len(content)} is larger than max_output_length {self.max_output_length}")

        data = [{"_id":str(uuid.uuid4()),
                "chat_name":self.chat_name,
                "role":role,
                "owner":owner,
                "content":self.search_tokenize(content),
                "raw_content":content,
                "auth_tag":"",
                "created_time":int(time.time()*1000),
                "chat_name_vector":self.emb(self.chat_name),
                "content_vector":self.emb(content)}]    

        self.retrieval.build_from_dicts(self.retrieval_cluster,self.retrieval_db,"user_memory",data)

    def get_conversations(self,owner:str, chat_name:str,limit=1000)->List[Dict[str,Any]]:
        docs = self.retrieval.filter(self.retrieval_cluster,
                        [SearchQuery(self.retrieval_db,"user_memory",
                                     filters={"and":[self._owner_filter(),{"field":"chat_name","value":chat_name}]},
                                     sorts=[{"created_time":"desc"}],
                                    keyword=None,fields=["chat_name"],
                                    vector=[],vectorField=None,
                                    limit=limit)])
        sorted_docs = sorted(docs[0:limit],key=lambda x:x["created_time"],reverse=False)
        return sorted_docs
    
    def get_conversations_as_history(self,limit=1000)->List[LLMHistoryItem]:
        chat_history = self.get_conversations(self.owner,self.chat_name,limit=limit)        
        chat_history = [LLMHistoryItem(item["role"],item["raw_content"]) for item in chat_history]
        return chat_history    


    def save_text_content(self,owner:str,title:str,content:str,url:str,auth_tag:str=""):

        if not self.retrieval:
            raise Exception("retrieval is not setup")
                        
        text_content = [{"_id":str(uuid.uuid4()),
            "title":self.search_tokenize(title),
            "content":self.search_tokenize(content),
            "owner":owner,
            "raw_content":content,
            "url":url,
            "auth_tag":self.search_tokenize(auth_tag),
            "title_vector":self.emb(title),
            "content_vector":self.emb(content)
            }]
        self.retrieval.build_from_dicts(self.retrieval_cluster,self.retrieval_db,"text_content",text_content)
        
        content_chunks= self.llm.apply_sql_func('''select llm_split(value,array(",","。","\n"),1600) as value ''',[{"value":content}])["value"]
        
        text_content_chunks = [{"_id":str(uuid.uuid4()),
            "doc_id":text_content[0]["_id"],
            "owner":owner,
            "chunk":self.search_tokenize(item["content"]),
            "raw_chunk":item["content"],
            "chunk_vector":self.emb(item["content"])
            } for item in content_chunks]
        
        self.retrieval.build_from_dicts(self.retrieval_cluster,self.retrieval_db,"text_content_chunk",text_content_chunks)

    def set_data_analysis_mode(self,mode:DataAnalysisMode):
        self.data_analysis_mode = mode
        return self
    
    def _owner_filter(self):
        return {"field":"owner","value":self.owner}
    

            
    def search_content_chunks(self,q:str,limit:int=4,return_json:bool=True):   
        docs = self.retrieval.search(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content_chunk",
                                         filters={"and":[self._owner_filter()]},
                                        keyword=self.search_tokenize(q),fields=["chunk"],
                                        vector=self.emb(q),vectorField="chunk_vector",
                                        limit=limit)])

        if return_json:
            context = json.dumps([{"content":x["raw_chunk"]} for x in docs],ensure_ascii=False,indent=4)    
            return context 
        else:
            return docs
        
    def get_doc(self,doc_id:str):
        docs = self.retrieval.search(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content",
                                         filters={"and":[self._owner_filter()]},
                                        keyword=doc_id,fields=["_id"],
                                        vector=[],vectorField=None,
                                        limit=1)])
        return docs[0] if docs else None
    
    def get_doc_by_url(self,url:str):
        docs = self.retrieval.search(self.retrieval_cluster,
                            [SearchQuery(self.retrieval_db,"text_content",
                                         filters={"and":[self._owner_filter()]},
                                        keyword=url,fields=["url"],
                                        vector=[],vectorField=None,
                                        limit=1)])
        return docs[0] if docs else None
                
        
    def search_memory(self,chat_name:str, q:str,limit:int=4,return_json:bool=True):
        docs = self.retrieval.search(self.retrieval_cluster,
                        [SearchQuery(self.retrieval_db,"user_memory",
                                     filters={"and":[self._owner_filter()]},
                                    keyword=chat_name,fields=["chat_name"],
                                    vector=self.emb(q),vectorField="content_vector",
                                    limit=1000)])
        docs = [doc for doc in docs if doc["role"] == "user" and doc["chat_name"] == chat_name]
        if return_json:
            context = json.dumps([{"content":x["raw_chunk"]} for x in docs[0:limit]],ensure_ascii=False,indent=4)    
            return context 
        else:
            return docs[0:limit]
        
            
    def analyze(self,prompt:str,max_try_times=10, **config)-> ExecuteCodeResponse:
        if self.data_analysis_mode == DataAnalysisMode.data_analysis:
            return self.data_analyze(prompt,max_try_times,**config)
        elif self.data_analysis_mode == DataAnalysisMode.text_analysis:
            return self.text_analyze(prompt,max_try_times,**config)
        
    def text_analyze(self,prompt:str,max_try_times=10,**config)-> ExecuteCodeResponse:
        recall_limit = 4
        if "recall_limit" in config or "limit" in config:
            recall_limit = config["recall_limit"] if "recall_limit" in config else config["limit"]

        memory_limit = 100
        if "memory_limit" in config:
            memory_limit = config["memory_limit"]  

        if "chunk_size" in config:
            chunk_size = config["chunk_size"]
        else:
            chunk_size = self.max_input_length - 600

        is_summary = utils.is_summary(self,prompt,self.role_mapping)
        if self.verbose:
            print(f'''
=============== Check Is Summary Requirement =================
------prompt------
{prompt}
------response------
is_summary: {is_summary}
''',flush=True)

        if is_summary:             
            doc = self.get_doc_by_url(self.file_path)
            raw_content = doc["raw_content"]
            multipe = len(raw_content) > chunk_size
            answer_chunk = ""
            if  multipe:
                for i in range(100):
                    start = i * chunk_size
                    end = (i+1) * chunk_size - len(answer_chunk)
                    if end < start or end > len(raw_content):
                        break
                    
                    if self.verbose:
                        print(f'''
=============== Summary Text =================
start: {start} end: {end} 
len_answer_chunk: {len(answer_chunk)}
answer_chunk: {answer_chunk}
''',flush=True)
                    if raw_content[start:end] == "":
                        break
                                                        
                    p = PROMPTS.prompt_sumarization(answer_chunk,raw_content[start:end],prompt)   
                    answer_chunk = self.llm.chat(None,request=
                                                 LLMRequest.build(instruction=p,
                                                                   max_length=self.max_length,
                                                                   temperature=self.tempraure,
                                                                   role_mapping=self.role_mapping)
                                                 )[0].output 
            else:
                p = PROMPTS.prompt_sumarization("",raw_content,prompt)
                answer_chunk = self.llm.chat(None,request=LLMRequest.build(instruction=p,
                                                                   max_length=self.max_length,
                                                                   temperature=self.tempraure,
                                                                   role_mapping=self.role_mapping))[0].output
                if self.verbose:
                    print(f'''
=============== Summary Text =================
------prompt------
{p}
------response------
{answer_chunk}
''',flush=True) 
            if self.keep_conversation:    
                self.save_conversation(self.owner,Role.User,prompt)
                self.save_conversation(self.owner,Role.Assistant,answer_chunk)     
            return ExecuteCodeResponse(0,answer_chunk,"",p,{}) 
        
        content = self.search_content_chunks(q=prompt,limit=recall_limit,return_json=True)
        p1 = PROMPTS.prompt_analyze_text(content,prompt)
        chat_history = self.get_conversations_as_history(limit=memory_limit) 
        v1 = self.llm.chat(None,request=LLMRequest(instruction=p1,max_length=self.max_length,
                                                                   temperature=self.tempraure,extra_params=LLMRequestExtra(history=chat_history,**self.role_mapping)))[0].output
        p1_len = len(p1)
        if self.verbose:
            print(f'''
=============== Analyze Text =================
------prompt------
len:{p1_len}
{p1}
------response------
{v1}
''',flush=True)
            
        if self.keep_conversation:
            self.save_conversation(self.owner,Role.User,prompt)
            self.save_conversation(self.owner,Role.Assistant,v1) 
        return ExecuteCodeResponse(0,v1,"",p1,{})

    def data_analyze(self,prompt:str,max_try_times=10,**config)-> ExecuteCodeResponse:

        memory_limit = 100
        if "memory_limit" in config:
            memory_limit = config["memory_limit"] 
        # I want you to act as a data scientist and code for me. I have a dataset of [describe dataset]. 
        # Please write code for data visualisation and exploration.  
        # I want you to act as an academic. Please summarise the paper [...] in simple terms in one paragraph.        
        if not self.loaded_successfully:            
            raw_preview_file_prompt=PROMPTS.prompt_preview_file(file_path=self.file_path)
            
            preview_file_prompt = self.llm._generate_ins(LLMRequest(instruction=raw_preview_file_prompt,max_length=self.max_length,
                                                                   temperature=self.tempraure,extra_params=LLMRequestExtra(**self.role_mapping)))
            response = self.try_execute_code_until_resolved(prompt=preview_file_prompt,
                                                            raw_prompt=raw_preview_file_prompt,
                                                            target_names={"loaded_successfully":True,"file_preview":None},
                                                            max_try_times=max_try_times)
            
            if self.verbose:
                print(f'''
=============== Preview Data File {self.file_path} ===============
------prompt------                  
{preview_file_prompt}

------response------
Success: {response.status == 0 and  response.variables["loaded_successfully"] == True}                                   

''',flush=True)
                        
            if response.status != 0 or not response.variables["loaded_successfully"]:
                raise Exception(f'''Failed to load the file {self.file_path}. 
The code is:

```python
{response.code}
```

The response is:

```text
{response}
```        
''')
            else:                        
                self.file_preview = response.variables["file_preview"].to_csv(index=False)    
                self.loaded_successfully = True
                # keep this message in the sandbox
                sandbox = self.get_sandbox(f"CodeSandbox-{self.sandbox_suffix}")
                sandbox.set_value.remote("file_preview",self.file_preview)
                sandbox.set_value.remote("loaded_successfully",self.loaded_successfully)
        
        preview_csv = self.file_preview
        
        need_code = utils.should_generate_code_to_response(self,prompt,self.role_mapping)

        if self.verbose:
            print(f'''
=============== Check Need Code ===============
------prompt------
{prompt}

------response------
{need_code}                                   

''',flush=True)

        if not need_code:
            no_code_prompt=PROMPTS.prompt_no_need_code(file_path=self.file_path,prompt=prompt,preview_csv=preview_csv)
            # self.llm.chat(None,request=no_code_prompt)[0].output,"",no_code_prompt
            
            chat_history = self.get_conversations_as_history(limit=memory_limit)            

            r = self.llm.chat(None,request=LLMRequest(instruction=no_code_prompt,max_length=self.max_length,
                                                                   temperature=self.tempraure,extra_params=LLMRequestExtra(history=chat_history,**self.role_mapping)))[0].output
            
            if self.keep_conversation:
                self.save_conversation(self.owner,Role.User,prompt)
                self.save_conversation(self.owner,Role.Assistant,r)

            return ExecuteCodeResponse(
                status=0,output=r,
                variables={},code="",prompt=no_code_prompt
            )
        
        is_visualization = utils.is_visualization(self,prompt,self.role_mapping)
        visualization_prompt = "" if not is_visualization else PROMPTS.PROMPT_VISUALIZATION

        if self.verbose:
            print(f'''
=============== Check Is Visualization Requirement ===============
------prompt------                  
{prompt}

------response------
{is_visualization}                                   

''',flush=True)

        analyze_prompt = PROMPTS.prompt_analysis_data_with_visualization(file_path=self.file_path,
                                                                         visualization_prompt=visualization_prompt,
                                                                          preview_csv=preview_csv
                                                                         )
        chat_history = self.get_conversations_as_history(limit=memory_limit)                 
        
        # final_prompt = self.llm.generate_instruction_from_history(analyze_prompt+prompt,chat_history,self.role_mapping)
        final_prompt = self.llm._generate_ins(LLMRequest(instruction=analyze_prompt+prompt,max_length=self.max_length,
                                                                   temperature=self.tempraure,
                                                         extra_params=LLMRequestExtra(history=chat_history,**self.role_mapping)));    
        
        response = self.try_execute_code_until_resolved(prompt=final_prompt,
                                                        raw_prompt=analyze_prompt+prompt,
                                                         target_names={"image_base64":None},
                                                         max_try_times=max_try_times,
                                                         skip_check_target_names= not is_visualization
                                                         )
        if response.status != 0:
            raise Exception(f'''
Failed to analyze {self.file_path}.

The prompt is:

```text
{response.prompt}
```

The code is:

```python
{response.code}
```

The output is:

```text
{response.output}
```

variables:

```text
{list(response.variables.keys())}
```
''')   
        if self.keep_conversation:
            self.save_conversation(self.owner,Role.User,prompt)
            self.save_conversation(self.owner,Role.Assistant,response.output)               
        return response
    

    def is_visualization_response(self,reseponse:ExecuteCodeResponse)->bool:
        return "image_base64" in reseponse.variables
                        

    def try_execute_code_until_resolved(self,prompt:str,
                                        raw_prompt:str=None,
                                        target_names:Dict[str,Any]={}, 
                                        max_try_times:int=3,
                                        skip_check_target_names:bool=False)->ExecuteCodeResponse:
        codes,cost =self.generate_code(prompt)
        code = codes[0][1]

        status,output,response = self.eval_code(code,target_names)        

        for i in range(max_try_times):
            if status != 0: 
                old_code = code 

                ## multi-lines start     
                improve_prompt = f'''Try to fix the following problems:
```
{output}
```
The origin requirements: {raw_prompt}
'''
                ## multi-lines finish

                improve_response,_ = self.improve_code(code=code,
                                                       objective=improve_prompt)            
                lang,code = code_utils.extract_code(improve_response)[0]

                ## multi-lines start
                if self.verbose:       
                    print(f'''
=========== Improving Code {i} Times =============

----------- Failed Reason -----------
{output}

----------- Failed Code  -------------
{old_code}

----------- Improved Code -----------
{code}

----------- prompt -----------
{improve_prompt}
''')   
                ## multi-lines finish
                                 
                status,output,response = self.eval_code(code,target_names)                                
            else:
                if not target_names or skip_check_target_names:
                    break

                success,msg = self.default_check_eval_repsonse(response,target_names)
                if success:
                    break    
                else:

                    old_code = code
                    improve_prompt = f"The origin requirements: {raw_prompt}\nAfter execute the code, {msg}.\n Try to fix this problem.\n"
                    improve_response,_ = self.improve_code(code=code,objective=improve_prompt)                    
                    lang,code = code_utils.extract_code(improve_response)[0]
                    
                    ## multi-lines start
                    if self.verbose:       
                        print(f'''
=========== Improving Code {i} Times =============

----------- Failed Reason -----------
{msg}

----------- Failed Code  -------------
{old_code}

----------- Improved Code -----------
{code}

----------- prompt -----------
{improve_prompt}
''')   
                    ## multi-lines finish


                    status,output,response = self.eval_code(code,target_names)            
        # status,response,code   
        return ExecuteCodeResponse(
            status=status,
            output=output,
            variables=response,
            code=code,
            prompt=prompt,
        )

    def get_target_names(self,prompt:str)->List[str]:
        self.llm.chat(None,request=LLMRequest(instruction=f'''Try to extract variables described in the following content:
```text                                     
{prompt}                                                                                            
```

and then output the variables in the following format:

```json
["a","b","c"]
```'''))     
    
    def improve_code(self,code:str=None,files:List[str]=None, objective:str=None,suggest_only=True, **config):
        """Improve the function to achieve the objective."""        
        # read the entire file into a str
        if code is None and files is None:
            raise Exception("code or files must be provided")
        
        final_code = ""
        if code is not None:
            final_code = code

        if files is not None:    
            for file_name in files:
                # read the entire file into a string
                with open(file_name, "r") as f:
                    file_string = f.read()
                final_code += f"""{file_name}:
{file_string}

"""     
        followup = "" if suggest_only else " followed by the improved code"    
        new_prompt = f'''Analyze the code in the following files and return a list of suggestions for improvement{followup}, to achieve the objective: '{objective}'.
{final_code}'''
        response = self.llm.chat(None, request=LLMRequest(instruction=new_prompt,**config))            
        return response[0].output, -1 
    
    def generate_assertions(self,definition: str, **config):
        prompt = f'''Given the signature and docstring, write the exactly same number of assertion(s) for the provided example(s) in the docstring, without assertion messages.

func signature:
{definition}
assertions:'''
        response = self.llm.chat(None, request=LLMRequest(instruction=prompt,**config))            
        assertions = response[0].output
        return assertions, -1
    
    def implement(self,
                    definition: str,
                    config: Dict[str,Any] = None,
                    assertions: Optional[Union[str, Callable[[str], Tuple[str, float]]]] = generate_assertions,
                ) -> Tuple[str, float]:
        """Implement a function from a definition.

        Args:
            definition (str): The function definition, including the signature and docstr.
            config (dict): The configuration for the API call.
            assertions (Optional, str or Callable): The assertion code which serves as a filter of the responses, or an assertion generator.

        Returns:
            str: The implementation.
            float: The cost of the implementation.
            int: The index of the configuration which generates the implementation.
        """
        # cost = 0
        
        # if callable(assertions):
        #     assertions, cost = assertions(definition)
        # assertion_filter = code_utils.PassAssertionFilter(assertions)
        response = self.llm.chat(None,request=LLMRequest(instruction=f'# Python 3{definition}',**config))
        # cost += assertion_filter.cost + 0
        return response[0].output

        # for i, config in enumerate(configs):
        #     response = oai.Completion.create({"definition": definition}, **config)
        #     cost += oai.Completion.cost(response)
        #     responses = oai.Completion.extract_text(response)
        #     metrics = eval_function_completions(responses, definition, assertions=assertions)
        #     assertions = metrics["assertions"]
        #     cost += metrics["gen_cost"]
        #     if metrics["succeed_assertions"] or i == len(configs) - 1:
        #         return responses[metrics["index_selected"]], cost, i

    def execute_code(self, code)->Tuple[int, str, str]:
        name = f"CodeSandbox-{self.sandbox_suffix}"
        
        sandbox = self.get_sandbox(name)           
        
        status,response,image = ray.get(sandbox.execute.remote(code))
        return status,response,image
    
    def get_sandbox_manager(self)->ClientActorHandle:
        name = "SANDBOX_MANAGER"
        manager = None
        try:
            manager = ray.get_actor(name)
        except Exception:              
            manager = ray.remote(SandboxManager).options(
                name=name, 
                lifetime="detached",               
                num_cpus=1,
                num_gpus=0
            ).remote()
        return manager 
    
    def get_sandbox(self,name:str)->ClientActorHandle:
        return ray.get(self.sandbox_manager.get_sandbox.remote(name) )

    def check_sandbox_exists(self,name:str)->bool:
        return ray.get(self.sandbox_manager.check_sandbox_exists.remote(name) )
    
    def get_or_create_sandbox(self,name:str)->ClientActorHandle:
        return ray.get(self.sandbox_manager.get_or_create_sandbox.remote(name
                                                                         ,self.file_path,self.file_ref,self.num_gpus,self.num_cpus))
            
    
    def eval_code(self, code,target_names:Dict[str,Any]={})->Tuple[int, str, str]:                
        name = f"CodeSandbox-{self.sandbox_suffix}"                
        sandbox = self.get_sandbox(name)        

        status,output,response = ray.get(sandbox.exec_capture_output.remote(code,target_names))            

        return status,output,response