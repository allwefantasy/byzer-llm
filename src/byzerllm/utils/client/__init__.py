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
from ..retrieval import ByzerRetrieval,TableSettings,SearchQuery
from .. import prompts as PROMPTS
import logging
import time
import math


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a enum for the role
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
    output: str
    input: str

@dataclasses.dataclass
class LLMRequestExtra:
    system_msg:str = "You are a helpful assistant. Think it over and answer the user question correctly."
    user_role:str = "User"
    assistant_role:str = "Assistant"
    history:List[LLMHistoryItem] = dataclasses.field(default_factory=list)
    


@dataclasses.dataclass
class LLMRequest:
    instruction: Union[str,List[str]]
    embedding: bool = False
    max_length: int = 4096
    top_p: float = 0.7
    temperature: float = 0.9
    extra_params: LLMRequestExtra = LLMRequestExtra()
    
    @classmethod
    def build(cls, instruction:str,max_length:int=4096,temperature:float=0.1,role_mapping:Dict[str,str]={}):
        return cls(instruction=instruction,max_length=max_length,temperature=temperature,extra_params=LLMRequestExtra(**role_mapping))
        

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
    DeepSpeed = "ray/deepspeed"

@dataclasses.dataclass
class ExecuteCodeResponse:
      status: int
      output: str      
      code: str
      prompt: str
      variables: Dict[str,Any]=dataclasses.field(default_factory=dict)

class ByzerLLM:
    def __init__(self,url:Optional[str]=None,**kwargs):
        self.url = url       
        self.default_sys_conf = {"pythonMode":"ray",
                         "maxConcurrency":1,
                         "num_gpus":1,
                         "masterMaxConcurrency":1000,
                         "workerMaxConcurrency":1,
                         "infer_backend":"transformers"
                         }
        self.sys_conf = self.default_sys_conf.copy()
        self.sql_model = "context" in globals()

        self.default_model_name = None

        if url is not None and self.sql_model:            
            v = globals()
            self.context = v["context"]
            self.ray_context = RayContext.connect(v, self.url, **kwargs)
        else:
            self.context = PythonContext(
                0,[],self.sys_conf
            ) 
            self.context.have_fetched = True
            self.ray_context = self.context.rayContext
    
    def setup_reset(self):
        self.sys_conf = self.default_sys_conf.copy()
        self.context.conf = self.sys_conf

    def setup_default_model_name(self,model_name:str)->'ByzerLLM':
        self.default_model_name = model_name
        return self    

    def setup(self,name:str, value:Any)->'ByzerLLM':
        self.sys_conf[name]=value
        # update the context conf
        self.context.conf = self.sys_conf
        return self
    
    def setup_infer_backend(self,backend:str)->'ByzerLLM':
        self.sys_conf["infer_backend"] = backend
        if backend == InferBackend.VLLM:            
            self.sys_conf["masterMaxConcurrency"] = 1000
        return self
    
    def setup_gpus_per_worker(self,num_gpus:int)->'ByzerLLM':
        self.sys_conf["num_gpus"] = num_gpus
        return self

    def setup_num_workers(self,num_workers:int)->'ByzerLLM':
        self.sys_conf["maxConcurrency"] = num_workers
        return self
    
    def raw_sft(self,train_params:Dict[str,Any]):                   
        model_type = train_params["pretrainedModelType"] .split("/")[-1]      
        train_module = importlib.import_module(f'byzerllm.{model_type}')
        sft_train = getattr(train_module,"sft_train")
        sft_train([],train_params,self.sys_conf)
            

    def raw_pretrain(self,train_params:Dict[str,Any]):                  
        model_type = train_params["pretrainedModelType"][-1]      
        train_module = importlib.import_module(f'byzerllm.{model_type}')
        sfft_train = getattr(train_module,"sfft_train")
        sfft_train([],train_params,self.sys_conf)

    def raw_merge_lora(self,train_params:Dict[str,Any]):                
        from byzerllm.utils.sft.merge_lora import merge_lora_to_base_model    
        merge_lora_to_base_model([],train_params,self.sys_conf) 

    def raw_deepspeed_to_huggingface(self,train_params:Dict[str,Any]):
        from byzerllm.utils.fulltune.pretrain.convert_to_transformers import convert
        convert(train_params,self.conf()) 

    def undeploy(self,udf_name:str):                  
        try:
            model = ray.get_actor(udf_name)
            ray.kill(model)        
        except ValueError:
            pass

    def generate_instruction_from_history(self,conversations:List[Dict[str,str]],role_mapping:Dict[str,str]={        
        "user_role":"User",        
        "assistant_role":"Assistant",
    }):
        
        new_his = []    
        for item in conversations:
            if item["role"] == "system":
                new_his.append(item["content"])
                continue        
            k = item['role']+"_role"            
            new_his.append(f"{role_mapping[k]}:{item['content']}")            
        
        if conversations[-1]["role"] == "user":            
            new_his.append(f"{role_mapping['assistant_role']}:")

        fin_ins = "\n".join(new_his)
        return fin_ins     

    def is_model_exist(self,udf_name:str)->bool:
        try:
            ray.get_actor(udf_name)
            return True
        except Exception as inst:
            return False    
               

    def deploy(self,model_path:str,
               pretrained_model_type:str,
               udf_name:str,
               infer_params:Dict[str,Any]):        
        from byzerllm import common_init_model
        self.setup("UDF_CLIENT",udf_name)
        model_type = pretrained_model_type
        
        if pretrained_model_type.startswith("saas/"):
            model_type = pretrained_model_type.split("/")[-1]                       
            infer_module = importlib.import_module(f'byzerllm.saas.{model_type}',"CustomSaasAPI")
            from byzerllm.utils.text_generator import simple_predict_func
            def init_model(model_refs: List[ClientObjectRef], conf: Dict[str, str]) -> Any:
                from byzerllm import consume_model
                consume_model(conf)                
                infer = infer_module(infer_params)
                return (infer,None)
            UDFBuilder.build(self.ray_context,init_model,simple_predict_func)
            return 


        if pretrained_model_type == "bark":
            from byzerllm.bark.bark_voice import build_void_infer, ZH_SPEAKER, EN_SPEAKER            
            def init_model(model_refs: List[ClientObjectRef], conf: Dict[str, str]) -> Any:
                infer = build_void_infer(
                model_dir=model_path,
                tokenizer_dir=f"{model_path}/pretrained_tokenizer")
                return infer
            def predict_func(model,v):
                data = [json.loads(item) for item in v]
                results=[{"predict":model.text_to_voice(item["instruction"]).tolist(),"labels":""} for item in data]
                return {"value":[json.dumps(results,ensure_ascii=False,indent=4)]}
            UDFBuilder.build(self.ray_context,init_model,predict_func)
            return                
        
        
        if pretrained_model_type.startswith("custom/"):
            model_type = pretrained_model_type.split("/")[-1]

        predict_func = "simple_predict_func"
        if model_type == "chatglm2":
            predict_func = "chatglm_predict_func"

        infer_module = importlib.import_module(f'byzerllm.{model_type}')
        predict_module = importlib.import_module(f"byzerllm.utils.text_generator")
        
        def init_model(model_refs: List[ClientObjectRef], conf: Dict[str, str]) -> Any:
            common_init_model(model_refs,conf,model_path, is_load_from_local=True)
            model = infer_module.init_model(model_path,infer_params,conf)
            return model
        
        UDFBuilder.build(self.ray_context,init_model,getattr(predict_module,predict_func))


    def emb(self, model, request:LLMRequest ,extract_params:Dict[str,Any]={})->List[List[float]]:
        
        if not model and not self.default_model_name:
            raise Exception("model name is required")
        
        if not model:
            model = self.default_model_name

        if isinstance(request,list):
            request = LLMRequest(instruction=request)

        if isinstance(request.instruction,str):
            v = [{
            "instruction":request.instruction,
            "embedding":True,
            "max_length":request.max_length,
            "top_p":request.top_p,
            "temperature":request.temperature,
            ** request.extra_params.__dict__,
            ** extract_params}] 
        else: 
            v = [{
            "instruction":x,
            "embedding":True,
            "max_length":request.max_length,
            "top_p":request.top_p,
            "temperature":request.temperature,
            ** request.extra_params.__dict__,
            ** extract_params} for x in request.instruction]    
        res = self._query(model,v) 
      
        return [LLMResponse(output=item["predict"],input=item["input"]) for item in res]
            
    def _generate_ins(self,request:LLMRequest):
         if not request.extra_params.user_role:
             return request.instruction
         
         conversations = [{"role":"system","content":request.extra_params.system_msg}]
         conversations += [{"role":item.role,"content":item.content} for item in request.extra_params.history]
         conversations += [{
                        "role":"user",
                        "content":request.instruction
                 }]
         
         final_ins = self.generate_instruction_from_history(conversations,                 
                 {
                    "user_role":request.extra_params.user_role,
                    "assistant_role":request.extra_params.assistant_role,
                    "system_msg":request.extra_params.system_msg
             })                      
             
         return final_ins
    
    def raw_chat(self,model,request:Union[LLMRequest,str],extract_params:Dict[str,Any]={})->List[LLMResponse]:
        if isinstance(request,str): 
            request = LLMRequest(instruction=request, extra_params=LLMRequestExtra(user_role=None))
        request.extra_params.user_role = None    
        return self.chat(model,request,extract_params)

    def chat(self,model,request:Union[LLMRequest,str],extract_params:Dict[str,Any]={})->List[LLMResponse]:
        if not model and not self.default_model_name:
            raise Exception("model name is required")
        
        if not model:
            model = self.default_model_name
        
        if isinstance(request,str): 
            request = LLMRequest(instruction=request)

        if isinstance(request.instruction,str):
            params = {**request.extra_params.__dict__,**extract_params}
            if "history" in params:
                del params["history"]

            v = [{
            "instruction":self._generate_ins(request),
            "max_length":request.max_length,
            "top_p":request.top_p,
            "temperature":request.temperature,            
             ** params}] 
        else: 
            v = []
            for x in request.instruction:
                
                new_request = LLMRequest(instruction=x,extra_params=request.extra_params,
                                         embedding=request.embedding,max_length=request.max_length,top_p=request.top_p,
                                         temperature=request.temperature
                                         )
                
                params = {**new_request.extra_params.__dict__,**extract_params}
                if "history" in params:
                    del params["history"]

                v.append({
                "instruction":self._generate_ins(new_request), 
                "max_length":request.max_length,
                "top_p":request.top_p,
                "temperature":request.temperature,           
                ** params
                })
        res = self._query(model,v) 
        return [LLMResponse(output=item["predict"],input=item["input"]) for item in res]
    
    def apply_sql_func(self,sql:str,data:List[Dict[str,Any]],owner:str="admin",url:str="http://127.0.0.1:9003/model/predict"):
        res = self._rest_byzer_engine(sql,data,owner,url)
        return res
                   
    def _rest_byzer_engine(self, sql:str,table:List[Dict[str,Any]],owner:str,url:str):
        import requests
        import json
        data = {
                'sessionPerUser': 'true',
                'sessionPerRequest': 'true',
                'owner': owner,
                'dataType': 'row',
                'sql': sql,
                'data': json.dumps(table,ensure_ascii=False)
            }
        response = requests.post(url, data=data)
        
        if response.status_code != 200:
            raise Exception(f"{self.url} status:{response.status_code} content: {response.text} request: json/{json.dumps(data,ensure_ascii=False)}")
        res = json.loads(response.text)        
        return res[0]

    def _query(self, model:str, input_value:List[Dict[str,Any]]):
        udf_master = ray.get_actor(model)
        new_input_value = [json.dumps(x,ensure_ascii=False) for x in input_value]
      
        try:
            [index, worker] = ray.get(udf_master.get.remote())
            res = ray.get(worker.async_apply.remote(new_input_value))            
            return json.loads(res["value"][0])
        except Exception as inst:
            raise inst
        finally:
            ray.get(udf_master.give_back.remote(index)) 

class CodeSandbox:
    def __init__(self,file_path:str,file_ref) -> None:
        self.file_ref = file_ref
        self.file_path = file_path
        if self.file_ref:
            if isinstance(self.file_ref, str):
                content = self.file_ref
            else:
                content = ray.get(self.file_ref)
            with open(self.file_path, "w") as f:
                f.write(content)         

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
                    "user_role":"User",
                    "assistant_role": "Assistant",
                    "system_msg":"You are a helpful assistant. Think it over and answer the user question correctly."
                    }, 
                 max_length:int=8024,   
                 tempraure:float=0.1,
                 max_input_length=1024*24,
                 verbose:bool=False, 
                 keep_conversation:bool=True,             
                 num_gpus=0, num_cpus=1) -> None:
        self.llm = llm
        self.retrieval = retrieval
        self.data_analysis_mode = data_analysis_mode
        self.max_input_length = max_input_length
        self.use_shared_disk = use_shared_disk
        self.sandbox = None
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

        self.sandbox_suffix = str(uuid.uuid4())                

        self.owner = owner
        self.chat_name = chat_name
        
        if self.owner is None:
            self.owner = self.sandbox_suffix            
        
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus

        
        
        if self.file_path and not self.use_shared_disk  and self.data_analysis_mode == DataAnalysisMode.data_analysis:
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            new_base_name = self.sandbox_suffix + ext
            dir_name = os.path.dirname(file_path)
            new_file_path = os.path.join(dir_name, new_base_name)

            logger.info(f"use_shared_disk: {self.use_shared_disk} file_path: {self.file_path} new_file_path: {new_file_path}")
            self.file_ref = ray.put(open(self.file_path).read())
            self.file_path = new_file_path

        if self.file_path and self.data_analysis_mode == DataAnalysisMode.text_analysis:
            content = open(self.file_path).read()
            self.save_text_content(title="noops",owner=self.owner,content=content,url=self.file_path)


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

        if self.chat_name is None:
            self.chat_name = content[0:10]   

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
        sorted_docs = sorted(docs[0:,limit],key=lambda x:x["created_time"],reverse=False)
        return sorted_docs
    
    def get_conversations_as_history(self,limit=1000)->List[LLMHistoryItem]:
        chat_history = self.get_conversations(self.owner,self.chat_name,limit=limit)        
        chat_history = [LLMHistoryItem(item["role"],item["raw_content"]) for item in chat_history]
        return chat_history    


    def save_text_content(self,owner:str,title:str,content:str,url:str,auth_tag:str=""):

        if not self.retrieval:
            raise Exception("retrieval is not setup")
                

        if not self.retrieval.check_table_exists(self.retrieval_cluster,self.retrieval_db,"text_content"):
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

        if utils.is_summary(self,prompt,self.role_mapping): 
            doc = self.get_doc_by_url(self.file_path)
            raw_content = doc["raw_content"]
            multipe = len(raw_content) / self.max_input_length
            answer_chunk = ""
            if  multipe > 1:
                for i in range(math.ceil(multipe)):
                    start = i * self.max_input_length
                    end = (i+1) * self.max_input_length
                    print(f'''start: {start} end: {end} answer_chunk: {answer_chunk}''',flush=True)
                    if raw_content[start:end] == "":
                        break
                    
                    p = f'''                
please try to summarize the following text:

{answer_chunk}
{raw_content[start:end]}

Finally, please try to match the following requirements:

```
{prompt}
```
'''                    
                    answer_chunk = self.llm.chat(None,request=
                                                 LLMRequest.build(instruction=p,
                                                                   max_length=self.max_length,
                                                                   temperature=self.tempraure,
                                                                   role_mapping=self.role_mapping)
                                                 )[0].output 
            else:
                p = f'''                
please try to summarize the following text:

{raw_content}

Finally, please try to match the following requirements:

```
{prompt}
```
'''
                answer_chunk = self.llm.chat(None,request=LLMRequest.build(instruction=p,
                                                                   max_length=self.max_length,
                                                                   temperature=self.tempraure,
                                                                   role_mapping=self.role_mapping))[0].output 
            if self.keep_conversation:    
                self.save_conversation(self.owner,Role.User,prompt)
                self.save_conversation(self.owner,Role.Assistant,answer_chunk)     
            return ExecuteCodeResponse(0,answer_chunk,"",p,{}) 
        
        content = self.search_content_chunks(q=prompt,limit=recall_limit,return_json=True)
        p1 = f'''
We have the following json format data:

{content}

Try to answer quession according the json format data we provided above.
the question is:

{prompt}
'''
        chat_history = self.get_conversations_as_history(limit=memory_limit) 
        v1 = self.llm.chat(None,request=LLMRequest(instruction=p1,max_length=self.max_length,
                                                                   temperature=self.tempraure,extra_params=LLMRequestExtra(history=chat_history,**self.role_mapping)))[0].output
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
                self.file_preview = response.variables["file_preview"]    
                self.loaded_successfully = True
        
        preview_csv = self.file_preview.to_csv(index=False)                
        
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
            no_code_prompt=PROMPTS.prompt_no_need_code(file_path=self.file_path,sprompt=prompt,preview_csv=preview_csv)
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
        if self.sandbox is None:
            self.sandbox = ray.remote(CodeSandbox).options(
                name="CodeSandbox",                
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus
            ).remote()
        status,response,image = ray.get(self.sandbox.execute.remote(code))
        return status,response,image
    
    def eval_code(self, code,target_names:Dict[str,Any]={})->Tuple[int, str, str]:        
        if self.sandbox is None:
            self.sandbox = ray.remote(CodeSandbox).options(
                name=f"CodeSandbox-{self.sandbox_suffix}",                
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus
            ).remote(self.file_path,self.file_ref)

        status,output,response = ray.get(self.sandbox.exec_capture_output.remote(code,target_names))            

        return status,output,response
            




            