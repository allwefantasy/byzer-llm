from pyjava import PythonContext,RayContext
from typing import Dict,Any,List,Optional,Union,Tuple,Callable,Annotated
from pyjava.udf import UDFBuilder
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from byzerllm.utils.client import code_utils 
from byzerllm.utils import function_calling_format,response_class_format,response_class_format_after_chat,FunctionCallList
from langchain.prompts import PromptTemplate
import json
import dataclasses
import importlib  
import logging
import time
import asyncio
import pydantic


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
    DeepSpeed = "ray/deepspeed"

@dataclasses.dataclass
class ExecuteCodeResponse:
      status: int
      output: str      
      code: str
      prompt: str
      variables: Dict[str,Any]=dataclasses.field(default_factory=dict)

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
                        generation_config={},                  
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
        
        self.verbose = kwargs.get("verbose",False)
        
        self.force_skip_context_length_check = False
        if "force_skip_context_length_check" in kwargs:
            self.force_skip_context_length_check = kwargs["force_skip_context_length_check"]
        
        self.mapping_max_input_length = {}
        self.mapping_max_output_length = {}
        self.mapping_max_model_length = {}        
        self.mapping_role_mapping = {}
        self.mapping_extra_generation_params = {}
        self.mapping_clean_func = {}
   
        self.mapping_function_calling_format_func = {}
        self.mapping_response_class_format_func = {}
        self.mapping_response_class_format_after_chat_func = {}

        self.meta_cache = {}

        self.byzer_engine_url = None
        if "byzer_engine_url" in kwargs:
            self.byzer_engine_url = kwargs["byzer_engine_url"]  

        self.default_max_output_length = 1024
        if "default_max_output_length" in kwargs:
            self.default_max_output_length = kwargs["default_max_output_length"]   
        

        self.default_model_name = None
        self.default_emb_model_name = None
        self.default_role_mapping = {
                    "user_role":"User:",
                    "assistant_role": "Assistant:",
                    "system_msg":"You are a helpful assistant. Think it over and answer the user question correctly."
                    }

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

    def setup_default_emb_model_name(self,model_name:str)->'ByzerLLM':
        self.default_emb_model_name = model_name
        return self    

    def setup(self,name:str, value:Any)->'ByzerLLM':
        self.sys_conf[name]=value
        # update the context conf
        self.context.conf = self.sys_conf
        return self

    def setup_function_calling_format_func(self,model:str,func)->'ByzerLLM':
        self.mapping_function_calling_format_func[model] = func
        return self

    def setup_response_class_format_func(self,model:str,func)->'ByzerLLM':
        self.mapping_response_class_format_func[model] = func
        return self

    def setup_response_class_format_after_chat_func(self,model:str,func)->'ByzerLLM':
        self.mapping_response_class_format_after_chat_func[model] = func
        return self    
    
    def setup_infer_backend(self,backend:str)->'ByzerLLM':
        self.sys_conf["infer_backend"] = backend
        
        if backend == InferBackend.VLLM or backend == InferBackend.DeepSpeed:            
            self.sys_conf["masterMaxConcurrency"] = 1000
            self.sys_conf["workerMaxConcurrency"] = 100
        
        if backend == InferBackend.Transformers:
            self.sys_conf["masterMaxConcurrency"] = 1000
            self.sys_conf["workerMaxConcurrency"] = 1

        return self
    
    def setup_gpus_per_worker(self,num_gpus:int)->'ByzerLLM':
        self.sys_conf["num_gpus"] = num_gpus
        return self
    
    def setup_worker_concurrency(self,num:int)->'ByzerLLM':        
        self.sys_conf["workerMaxConcurrency"] = num
        return self        

    def setup_num_workers(self,num_workers:int)->'ByzerLLM':
        self.sys_conf["maxConcurrency"] = num_workers
        return self
    
    def setup_max_model_length(self,model:str,max_model_length:int)->'ByzerLLM':
        self.mapping_max_model_length[model] = max_model_length
        return self
    
    def setup_max_input_length(self,model:str,max_input_length:int)->'ByzerLLM':
        self.mapping_max_input_length[model] = max_input_length
        return self
    
    def setup_max_output_length(self,model:str, max_output_length:int)->'ByzerLLM':
        self.mapping_max_output_length[model] = max_output_length
        return self
    
    def setup_role_mapping(self,model:str,role_mapping:Dict[str,str])->'ByzerLLM':
        self.mapping_role_mapping[model] = role_mapping
        return self
    
    def setup_extra_generation_params(self,model:str,extra_generation_params:Dict[str,Any])->'ByzerLLM':
        self.mapping_extra_generation_params[model] = extra_generation_params
        return self
    
    def generate_llm_template(self,example:str,llm_template:Optional[str]=None)->Template:
        from byzerllm.utils.client.templates import LLM_TEMPALTE
        template = llm_template if llm_template else LLM_TEMPALTE
        m = template.replace("{example}",example)
        self.setup_max_output_length("chat",3000)
        t = self.chat_oai(conversations=[{"role":"user","content":m}])
        [(_,code)] = code_utils.extract_code(t[0].output)
        exec(code)
        return tpl()
    
    def setup_template(self,model:str,template:Union[Template,str])->'ByzerLLM':
        self.mapping_role_mapping[model] = template.role_mapping
        self.mapping_extra_generation_params[model] = template.generation_config
        self.mapping_clean_func[model] = template.clean_func
        self.mapping_function_calling_format_func[model] = template.function_calling_format_func
        self.mapping_response_class_format_after_chat_func[model] = template.response_class_format_after_chat_func
        self.mapping_response_class_format_func[model] = template.response_class_format_func
        return self

    def setup_auto(self,model:Optional[str])->'ByzerLLM':
        if not model:
            model = self.default_model_name            
        meta = self.get_meta(model=model)  
        # {'model_deploy_type': 'proprietary','backend':'ray/vllm','max_model_len': 8192, 'architectures': ['QWenLMHeadModel']}  
        if meta.get("model_deploy_type",None) != "proprietary":
           logger.info(f"model({model}) is not proprietary, skip auto setup")
           return self
        
        if "architectures" in meta:
            
            if "QWenLMHeadModel" in meta["architectures"]:
                self.setup_template(model,Templates.qwen())
                if "max_model_len" in meta:
                    self.setup_max_model_length(model,meta["max_model_len"])                        

        return self        


    def sft(self,sft_name:str,
            local_data_dir_path:str,
            local_model_path:str,
            local_stage_path:str,
            pretrained_model_type:str,            
            num_cpus:int,
            num_gpus:int,
            detached:bool=True,
            json_config:str="{}",
            model_params:Dict[str,Any]={},
            **kwargs
            ):
        '''
        finetune a pretrained model

        Args:
            sft_name (str): the uniq name of this finetune task
            local_data_dir_path (str): the local data dir path, which should contains `data.jsonl` file
            local_model_path (str): the local model path, which should contains `config.json` file
            local_stage_path (str): the local stage path which store the temp data and model
            pretrained_model_type (str): the pretrained model type, e.g. "sft/llama2","sft/baichuan"
            num_cpus (int): the number of cpus
            num_gpus (int): the number of gpus
            detached (bool, optional): whether to run this task in detached mode. Defaults to True.
            json_config (str, optional): the json config string. Defaults to "{}".
            model_params (Dict[str,Any], optional): the model params. Defaults to {}. The key should like this style `sft.int.logging_steps`, `sft.int.max_seq_length`
                                                    which contains the `sft` prefix and the type of the value.
        '''
        train_params = {}
        train_params["name"] = sft_name
        train_params["data_dir"] = local_data_dir_path
        train_params["localModelDir"] = local_model_path
        train_params["pretrainedModelType"] = pretrained_model_type
        train_params["config"] = json_config
        train_params["detached"] = "true" if detached else "false"
        train_params["localPathPrefix"] = local_stage_path
        
        for k,v in model_params.items():
            train_params[k] = v

        sys_conf = {}
        sys_conf["num_gpus"] = num_gpus
        sys_conf["num_cpus"] = num_cpus    

        r = self.raw_sft(train_params=train_params,sys_conf=sys_conf)
        if detached:
           return [i for i in r]
        return r
    
    def merge_lora(self,name:str,
                   local_model_path:str,
                   local_adpator_model_path:str,
                   local_target_path:str
                   ):
        train_params = {}
        train_params["name"] = name
        train_params["modelNameOrPath"] = local_model_path
        train_params["adapterNameOrPath"] = local_adpator_model_path
        train_params["savePath"] = local_target_path
        self.raw_merge_lora(train_params=train_params,sys_conf={})
        return local_target_path
    
    def pretrain(self,name:str,
            local_data_dir_path:str,
            local_model_path:str,
            local_stage_path:str,
            pretrained_model_type:str,            
            num_cpus:int,
            num_gpus:int,
            detached:bool=True,
            json_config:str="{}",
            model_params:Dict[str,Any]={},
            **kwargs):
        train_params = {}
        train_params["name"] = name
        train_params["localDataDir"] = local_data_dir_path
        train_params["localModelDir"] = local_model_path
        train_params["pretrainedModelType"] = pretrained_model_type
        train_params["deepspeedConfig"] = json_config
        train_params["detached"] = "true" if detached else "false"
        train_params["localPathPrefix"] = local_stage_path
        
        for k,v in model_params.items():
            train_params[k] = v

        sys_conf = {}
        sys_conf["num_gpus"] = num_gpus
        sys_conf["num_cpus"] = num_cpus    

        r = self.raw_pretrain(train_params=train_params,sys_conf=sys_conf)
        if detached:
           return [i for i in r]
        return r
    
    
    
    def raw_sft(self,train_params:Dict[str,Any],sys_conf:Dict[str,Any]={}):                   
        model_type = train_params["pretrainedModelType"] .split("/")[-1]              
        train_module =  importlib.import_module(f'byzerllm.{model_type}')
        return train_module.sft_train([],train_params,sys_conf)                
            

    def raw_pretrain(self,train_params:Dict[str,Any],sys_conf:Dict[str,Any]={}):                  
        model_type = train_params["pretrainedModelType"][-1]      
        train_module = importlib.import_module(f'byzerllm.{model_type}')        
        return train_module.sfft_train([],train_params,sys_conf)

    def raw_merge_lora(self,train_params:Dict[str,Any],sys_conf:Dict[str,Any]):                
        from byzerllm.utils.sft.merge_lora import merge_lora_to_base_model    
        merge_lora_to_base_model([],train_params,sys_conf) 

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
        "user_role":"User:",        
        "assistant_role":"Assistant:",
    }):
        
        new_his = []    
        for item in conversations:
            if item["role"] == "system":
                value = item["content"]
                if "system_msg_func" in role_mapping:
                    value = role_mapping["system_msg_func"](t=role_mapping["system_msg"],v=item["content"])
                new_his.append(value)
                continue
            
            if item["role"] == "user":
                value =  f"{role_mapping['user_role']}{item['content']}"
                if "user_role_func" in role_mapping:
                        value = role_mapping["user_role_func"](t=role_mapping["user_role"],v=item["content"])         
                new_his.append(value)  
            
            if item["role"] == "assistant":
                value =  f"{role_mapping['assistant_role']}{item['content']}"
                if "user_role_func" in role_mapping:
                        value = role_mapping["assistant_role_func"](t=role_mapping["assistant_role"],v=item["content"])         
                new_his.append(value)              
        
        if conversations[-1]["role"] == "user":            
            new_his.append(f"{role_mapping['assistant_role']}")

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

        infer_backend = self.sys_conf["infer_backend"]
        
        if infer_backend == InferBackend.VLLM or infer_backend == InferBackend.DeepSpeed:
            if pretrained_model_type != "custom/auto":
                raise ValueError(f"Backend({infer_backend}) is set. the pretrained_model_type should be `custom/auto`")

        model_type = pretrained_model_type
        
        if pretrained_model_type.startswith("saas/"):
            model_type = pretrained_model_type.split("/")[-1]                       
            
            infer_module = importlib.import_module(f'byzerllm.saas.{model_type}')
            from byzerllm.utils.text_generator import simple_predict_func
            
            def init_model(model_refs: List[ClientObjectRef], conf: Dict[str, str]) -> Any:
                from byzerllm import consume_model
                consume_model(conf)                
                infer = infer_module.CustomSaasAPI(infer_params)
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
        
        # we put in this place so it only take effect for private model
        self.mapping_max_output_length[udf_name]=1024

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
  
    def get_meta(self,model:str,llm_config:Dict[str,Any]={}):        
        if not model and not self.default_model_name:
            raise Exception("model name is required")
        
        if not model:
            model = self.default_model_name

        if model in self.meta_cache:
            return self.meta_cache[model]    

        default_config = self.mapping_extra_generation_params.get(model,{})

        v = [{"instruction":"","meta":True, **{**default_config,**llm_config} }]        
        res = self._query(model,v) 
        
        t = [LLMResponse(output=item["predict"],metadata=item.get("metadata",{}),input=item["input"]) for item in res]        
        
        res = {}
        if len(t) != 0 and len(t[0].output) != 0 :
            res = t[0].output[0]

        self.meta_cache[model] = res            
        return self.meta_cache[model]
        
    def tokenize(self,model:str,s:str,llm_config:Dict[str,Any]={})->List[str]:
        
        if not model and not self.default_model_name:
            raise Exception("model name is required")
        
        if not model:
            model = self.default_model_name

        default_config = self.mapping_extra_generation_params.get(model,{})

        v = [{"instruction":s,"tokenizer":True, **{**default_config,**llm_config} }]        
        res = self._query(model,v) 
        return [LLMResponse(output=item["predict"],metadata=item.get("metadata",{}),input=item["input"]) for item in res]
        
    def emb(self, model, request:LLMRequest ,extract_params:Dict[str,Any]={})->List[List[float]]:
        
        if not model and not self.default_model_name:
            raise Exception("model name is required")
        
        if not model:
            model = self.default_model_name

        default_config = self.mapping_extra_generation_params.get(model,{})            

        if isinstance(request,list):
            request = LLMRequest(instruction=request)

        if isinstance(request.instruction,str):
            v = [{
            "instruction":request.instruction,
            "embedding":True,
            "max_length":request.max_length,
            "top_p":request.top_p,
            "temperature":request.temperature,                                    
            ** default_config,           
            ** extract_params}] 
        else: 
            v = [{
            "instruction":x,
            "embedding":True,
            "max_length":request.max_length,
            "top_p":request.top_p,
            "temperature":request.temperature,            
            ** default_config, 
            ** extract_params} for x in request.instruction]    
        res = self._query(model,v) 
      
        return [LLMResponse(output=item["predict"],metadata=item.get("metadata",{}),input=item["input"]) for item in res]
            
    def _generate_ins(self,request:LLMRequest,role_mapping:Dict[str,str]):
         if not role_mapping["user_role"]:
             return request.instruction
         
         sys_msg = role_mapping["system_msg"]
         if "system_msg_func" in role_mapping:
             sys_msg = "You are a helpful assistant. Think it over and answer the user question correctly."
         
         conversations = [{"role":"system","content":sys_msg}]
         # conversations += [{"role":item.role,"content":item.content} for item in request.extra_params.history]
         
         conversations += self._to_openai_format(request=request)
         
         final_ins = self.generate_instruction_from_history(conversations,role_mapping)                      
             
         return final_ins
    
    def _to_openai_format(self,request:LLMRequest): 
        conversations = []
        if isinstance(request.instruction,str):       
            conversations += [{
                        "role":"user",
                        "content":request.instruction
                    }]
        else:
            conversations += [{
                        "role":"user",
                        "content":x
                    } for x in request.instruction]    
        return conversations

    def execute_function_calling(self,response:LLMResponse,tools:List[Callable])-> LLMFunctionCallResponse:            
        codes = code_utils.extract_code(response.output)
        
        r = LLMFunctionCallResponse(response=response,values=[],metadata={"reason":""})
        
        if len(codes) == 0:
            r.metadata["reason"] = "No json block found"
            return r 
        
        lang,code = codes[0]

        if lang != "json":
            r.metadata["reason"] = "No json block found"
            return r
        
        try:
            ms = FunctionCallList.parse_obj(json.loads(code))
        except Exception as inst:
            r.metadata["reason"] = str(inst)
            return r
                    
        _func_maps = dict([(t.__name__,t) for t in tools])
        
        try:
            for m in ms.tool_calls:        
                if m.function.name in _func_maps:
                    r.values.append(_func_maps[m.function.name](**m.function.arguments))
        except Exception as inst:
            r.metadata["reason"] = str(inst)            

        return r
    
    def execute_response_format(self,response:LLMResponse,response_class:pydantic.BaseModel):
        codes = code_utils.extract_code(response.output)
        
        r = LLMClassResponse(response=response,value=None,metadata={"reason":""})
        
        if len(codes) == 0:
            r.metadata["reason"] = "No json block found"
            return r 
        
        lang,code = codes[0]

        if lang != "json":
            r.metadata["reason"] = "No json block found"
            return r
        
        try:
            ms = response_class.parse_obj(json.loads(code))            
        except Exception as inst:
            r.metadata["reason"] = str(inst)
            return r                                       
        
        r.value=ms

        return r

    def chat_oai(self,
                 conversations,
                 tools:List[Callable]=[], 
                 tool_choice:Callable=None,
                 execute_tool:bool=False,  
                 response_class:Optional[pydantic.BaseModel] = None, 
                 response_after_chat:Optional[pydantic.BaseModel] = False,
                 model:Optional[str] = None,
                 role_mapping=None,llm_config:Dict[str,Any]={})->Union[List[LLMResponse],List[LLMFunctionCallResponse],List[LLMClassResponse]]:        
        
        if not self.default_model_name and not model:
            raise Exception("Use llm.setup_default_model_name to setup default model name or setup the model parameter")
        
        if not model:
            model = self.default_model_name
            
        if role_mapping is None:
            role_mapping = self.mapping_role_mapping.get(model, self.default_role_mapping)
        
        if response_class and (tools or tool_choice):
            raise Exception("function calling is enabled,response_class should be set.")
        
        # todo: try to cache the meta
        meta = self.get_meta(model=model)        
        is_saas_model =  meta.get("model_deploy_type",None) == "saas"

        
        last_message = conversations[-1]
        
        if tools or tool_choice:
            f = self.mapping_function_calling_format_func.get(model,function_calling_format)
            last_message["content"] = f(last_message["content"],tools,tool_choice)

        if response_class and not response_after_chat:
            f = self.mapping_response_class_format_func.get(model,response_class_format)
            last_message["content"] = f(last_message["content"],cls = response_class)
        
        if is_saas_model:
            final_ins = last_message["content"]
            history = conversations[:-1]
        else:
            final_ins = self.generate_instruction_from_history(conversations, role_mapping)         
            history = []

        default_config = self.mapping_extra_generation_params.get(model,{})
        v = [{"instruction":final_ins,"history":history,**default_config,**llm_config }]         
        res = self._query(model,v) 
        clean_func = self.mapping_clean_func.get(model,lambda s: s)        
        responses = [LLMResponse(output=clean_func(item["predict"]),metadata=item.get("metadata",{}),input=item["input"]) for item in res]        

        temp_result = responses    
        if response_class and response_after_chat: 
            temp_result = []
            f = self.mapping_response_class_format_after_chat_func.get(model,response_class_format_after_chat)
            for response in responses:
                new_conversations = conversations + [{
                                        "content":response.output,
                                        "role":"assistant"
                                    },{
                                        "content":f(response_class),
                                        "role":"user"
                                    }]
                temp_result.append(self.chat_oai(new_conversations,role_mapping=role_mapping,llm_config=llm_config)[0])            

        if response_class:
            final_result = []
            for response in temp_result:
                final_result.append(self.execute_response_format(response,response_class))
            return final_result    


        if not execute_tool:
            return responses
        
        if execute_tool:
            final_result = []
            for response in responses:
                final_result.append(self.execute_function_calling(response,tools))

            return final_result

        
    def stream_chat_oai(self,conversations, model:Optional[str]=None, role_mapping=None,llm_config:Dict[str,Any]={}): 
        
        if not model:
            model = self.default_model_name

        meta = self.get_meta(model=model)
        if not meta.get("support_stream",False):
            raise Exception(f"The model({model}) is not support stream chat for now.")

        v = self.chat_oai(conversations,model=model,role_mapping = role_mapping,llm_config={**llm_config,**{"generation.stream":True}})       
        request_id = v[0].metadata["request_id"]
        stream_server = v[0].metadata.get("stream_server","VLLM_STREAM_SERVER")
        server = ray.get_actor(stream_server)                        
        
        while True:                 
            final_output = ray.get(server.get_item.remote(request_id))
            if isinstance(final_output,str):
                time.sleep(0.01)
                continue
            
            if final_output is None:
                break
            
            text_outputs = final_output.outputs
            clean_func = self.mapping_clean_func.get(model,lambda s: s)
            generated_text = text_outputs[0].text                                
            yield clean_func(generated_text)

    async def async_stream_chat_oai(self,conversations,role_mapping=None,model:Optional[str]=None,llm_config:Dict[str,Any]={}): 
        
        if not model:
            model = self.default_model_name
        
        meta = self.get_meta(model=model)
        if not meta.get("support_stream",False):
            raise Exception(f"The model({model}) is not support stream chat for now.")    

        v = self.chat_oai(conversations,model=model,role_mapping=role_mapping,llm_config={**llm_config,**{"generation.stream":True}})       
        request_id = v[0].metadata["request_id"]
        stream_server = v[0].metadata.get("stream_server","VLLM_STREAM_SERVER")
        server = ray.get_actor(stream_server)                        
        
        while True:                 
            final_output = await server.get_item.remote(request_id)
            if isinstance(final_output,str):
                time.sleep(0.01)
                continue
            
            if final_output is None:
                break
            
            text_outputs = [output for output in final_output.outputs]
            clean_func = self.mapping_clean_func.get(model,lambda s: s)
            generated_text = text_outputs[0].text                                
            yield clean_func(generated_text)        
    

    def raw_chat(self,model,request:Union[LLMRequest,str],extract_params:Dict[str,Any]={})->List[LLMResponse]:
        if isinstance(request,str): 
            request = LLMRequest(instruction=request)

        return self.chat(model,request,extract_params)

    def chat(self,model,request:Union[LLMRequest,str],extract_params:Dict[str,Any]={})->List[LLMResponse]:
        if not model and not self.default_model_name:
            raise Exception("model name is required")
        
        if not model:
            model = self.default_model_name

        default_config = self.mapping_extra_generation_params.get(model,{})  
        
        default_role_mapping = self.mapping_role_mapping.get(model, self.default_role_mapping)  
        
        if isinstance(request,str): 
            request = LLMRequest(instruction=request)

        if isinstance(request.instruction,str):
            
            final_input = self._generate_ins(request,default_role_mapping)                         
            
            v = [{
            "instruction":final_input,
            "max_length":request.max_length,
            "top_p":request.top_p,
            "temperature":request.temperature,                       
             **default_config,**extract_params
             }] 
        else: 
            v = []
            for x in request.instruction:
                
                new_request = LLMRequest(instruction=x,
                                         embedding=request.embedding,max_length=request.max_length,top_p=request.top_p,
                                         temperature=request.temperature,
                                         )
                               
                final_input = self._generate_ins(new_request,default_role_mapping)                                    
                
                v.append({
                "instruction":final_input, 
                "max_length":request.max_length,
                "top_p":request.top_p,
                "temperature":request.temperature, 
                **default_config,          
                **extract_params
                })
        res = self._query(model,v) 
        clean_func = self.mapping_clean_func.get(model,lambda s: s)
        return [LLMResponse(output=clean_func(item["predict"]),metadata=item.get("metadata",{}),input=item["input"]) for item in res]
    
    def apply_sql_func(self,sql:str,data:List[Dict[str,Any]],owner:str="admin",url:str="http://127.0.0.1:9003/model/predict"):
        if self.byzer_engine_url and url == "http://127.0.0.1:9003/model/predict":
            url = self.byzer_engine_url
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

    def get_max_model_length(self,model:str):
        return self.mapping_max_model_length.get(model,None)

    def get_max_output_length(self,model:str):
        return self.mapping_max_output_length.get(model,self.default_max_output_length)

    def get_max_input_length(self,model:str):
        return self.mapping_max_input_length.get(model,None)        

    def _query(self, model:str, input_value:List[Dict[str,Any]]):  
        
        if not self.force_skip_context_length_check:
            for input in input_value:
                # if this is a embedding/tokenizer query ,skip            
                if input.get("embedding",False) or input.get("tokenizer",False):
                    continue            
                
                final_ins = input.get("instruction","")
                
                try:
                    input_size = len(self.tokenize(None,final_ins,{})[0].output[0])
                except Exception as inst:                
                    continue
                
                if self.get_max_input_length(model) and input_size > self.get_max_input_length(model):
                    raise Exception(f"input length {input_size} is larger than max_input_length {self.mapping_max_input_length[model]}")                
                
                max_output_length = self.get_max_output_length(model)

                if  self.get_max_model_length(model):                    
                    if input_size + max_output_length > self.get_max_model_length(model):
                        raise Exception(f"input_size ({input_size}) + max_output_length {max_output_length} is larget than model context length {self.mapping_max_model_length[model]}")                
                
                # dynamically update the max_length
                input["max_length"] = input_size + max_output_length


        udf_master = ray.get_actor(model)        
        new_input_value = [json.dumps(x,ensure_ascii=False) for x in input_value]
        if self.verbose:
            print(f"Send to model[{model}]:{new_input_value}")
      
        try:            
            [index, worker] = ray.get(udf_master.get.remote())                        
            res = ray.get(worker.async_apply.remote(new_input_value))                                    
            return json.loads(res["value"][0])
        except Exception as inst:
            raise inst
        finally:
            ray.get(udf_master.give_back.remote(index)) 

