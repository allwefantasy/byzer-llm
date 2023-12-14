from pyjava import PythonContext,RayContext
from typing import Dict,Any,List,Optional,Union,Tuple,Callable
from pyjava.udf import UDFBuilder
import ray
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import json
import dataclasses
import importlib  
import logging
import time
import asyncio


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
    metadata: Dict[str,Any] = dataclasses.field(default_factory=dict)

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
    def __init__(self,role_mapping:Dict[str,str],generation_config:Dict[str,Any],clean_func:Callable[[str],str]=lambda s: s ) -> None:
        self.role_mapping = role_mapping
        self.generation_config = generation_config
        self.clean_func = clean_func


class Templates:
    @staticmethod
    def qwen():
        def clean_func(v):            
            if "<|im_end|>" in v:
                v = v.split("<|im_end|>")[0]
            if "<|endoftext|>" in v:
                v = v.split("<|endoftext|>")[0]            
            return v    
        return Template({
                    "user_role":"<|im_start|>user\n",
                    "assistant_role": "<|im_end|>\n<|im_start|>assistant\n",
                    "system_msg":"<|im_start|>system\nYou are a helpful assistant. Think it over and answer the user question correctly.<|im_end|>"
                    },{
                        "generation.early_stopping":False,
                        "generation.repetition_penalty":1.1,
                        "generation.stop_token_ids":[151643,151645]},clean_func) 
         

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
    
    def setup_template(self,model:str,template:Template)->'ByzerLLM':
        self.mapping_role_mapping[model] = template.role_mapping
        self.mapping_extra_generation_params[model] = template.generation_config
        self.mapping_clean_func[model] = template.clean_func
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
        "user_role":"User:",        
        "assistant_role":"Assistant:",
    }):
        
        new_his = []    
        for item in conversations:
            if item["role"] == "system":
                new_his.append(item["content"])
                continue        
            k = item['role']+"_role"            
            new_his.append(f"{role_mapping[k]}{item['content']}")            
        
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

    def tokenize(self,model:str,s:str,llm_config:Dict[str,Any]={})->List[str]:
        
        if not model and not self.default_model_name:
            raise Exception("model name is required")
        
        if not model:
            model = self.default_model_name

        default_config = self.mapping_extra_generation_params.get(model,{})

        v = [{"instruction":s,"tokenizer":True, **{**default_config,**llm_config} }]        
        res = self._query(model,v) 
        return [LLMResponse(output=item["predict"],metadata=item["metadata"],input=item["input"]) for item in res]
        
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
      
        return [LLMResponse(output=item["predict"],metadata=item["metadata"],input=item["input"]) for item in res]
            
    def _generate_ins(self,request:LLMRequest,role_mapping:Dict[str,str]):
         if not role_mapping["user_role"]:
             return request.instruction
         
         conversations = [{"role":"system","content":role_mapping["system_msg"]}]
         conversations += [{"role":item.role,"content":item.content} for item in request.extra_params.history]
         
         conversations += [{
                        "role":"user",
                        "content":request.instruction
                 }]
         
         final_ins = self.generate_instruction_from_history(conversations,role_mapping)                      
             
         return final_ins
    
    def _to_openai_format(self,request:LLMRequest):
        conversations = [{"role":"system","content":request.extra_params.system_msg}]
        conversations += [{"role":item.role,"content":item.content} for item in request.extra_params.history]
        
        conversations += [{
                    "role":"user",
                    "content":request.instruction
                }]
        return conversations

    def chat_oai(self,conversations,role_mapping=None,**llm_config):        
        if role_mapping is None:
            role_mapping = self.mapping_role_mapping.get(self.default_model_name, self.default_role_mapping)
        
        final_ins = self.generate_instruction_from_history(conversations, role_mapping)          
        default_config = self.mapping_extra_generation_params.get(self.default_model_name,{})
        v = [{"instruction":final_ins,**default_config,**llm_config }]         
        res = self._query(self.default_model_name,v) 
        clean_func = self.mapping_clean_func.get(self.default_model_name,lambda s: s)
        return [LLMResponse(output=clean_func(item["predict"]),metadata=item["metadata"],input=item["input"]) for item in res]
        
    def stream_chat_oai(self,conversations,role_mapping=None,**llm_config): 
        v = self.chat_oai(conversations,role_mapping,**{**llm_config,**{"generation.stream":True}})       
        request_id = v[0].metadata["request_id"]
        server = ray.get_actor("VLLM_STREAM_SERVER")                
        
        while True:                 
            final_output = asyncio.run(server.get_item.remote(request_id))
            if isinstance(final_output,str):
                time.sleep(0.01)
                continue
            
            if final_output is None:
                break
            
            text_outputs = [output for output in final_output.outputs]
            clean_func = self.mapping_clean_func.get(self.default_model_name,lambda s: s)
            generated_text = text_outputs[0].text                                
            yield clean_func(generated_text)

    async def async_stream_chat_oai(self,conversations,role_mapping=None,**llm_config): 
        v = self.chat_oai(conversations,role_mapping,**{**llm_config,**{"generation.stream":True}})       
        request_id = v[0].metadata["request_id"]
        server = ray.get_actor("VLLM_STREAM_SERVER")                
        
        while True:                 
            final_output = await server.get_item.remote(request_id)
            if isinstance(final_output,str):
                time.sleep(0.01)
                continue
            
            if final_output is None:
                break
            
            text_outputs = [output for output in final_output.outputs]
            clean_func = self.mapping_clean_func.get(self.default_model_name,lambda s: s)
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
                
                new_request = LLMRequest(instruction=x,extra_params=request.extra_params,
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
        clean_func = self.mapping_clean_func.get(self.default_model_name,lambda s: s)
        return [LLMResponse(output=clean_func(item["predict"]),metadata=item["metadata"],input=item["input"]) for item in res]
    
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


def default_chat_wrapper(llm:"ByzerLLM",conversations: Optional[List[Dict]] = None,llm_config={}):
    return llm.chat_oai(conversations=conversations,**llm_config)

async def async_stream_qwen_chat_wrapper(llm:"ByzerLLM",conversations: Optional[List[Dict]] = None,llm_config={}):
    for conv in conversations:
        if conv["role"] == "system":
            if "<|im_start|>" not in conv["content"]:
                conv["content"] = "<|im_start|>system\n" + conv["content"] + "<|im_end|>"
            
    t = llm.async_stream_chat_oai(conversations=conversations,role_mapping={
                    "user_role":"<|im_start|>user\n",
                    "assistant_role": "<|im_end|>\n<|im_start|>assistant\n",
                    "system_msg":"<|im_start|>system\nYou are a helpful assistant. Think it over and answer the user question correctly.<|im_end|>"
                    },  **{**{
                        "max_length":1024*16,
                        "top_p":0.95,
                        "temperature":0.01,
                    },**llm_config,**{
                        "generation.early_stopping":False,
                        "generation.repetition_penalty":1.1,
                        "generation.stop_token_ids":[151643,151645]}})       
    return t

def stream_qwen_chat_wrapper(llm:"ByzerLLM",conversations: Optional[List[Dict]] = None,llm_config={}):
    for conv in conversations:
        if conv["role"] == "system":
            if "<|im_start|>" not in conv["content"]:
                conv["content"] = "<|im_start|>system\n" + conv["content"] + "<|im_end|>"
            
    t = llm.stream_chat_oai(conversations=conversations,role_mapping={
                    "user_role":"<|im_start|>user\n",
                    "assistant_role": "<|im_end|>\n<|im_start|>assistant\n",
                    "system_msg":"<|im_start|>system\nYou are a helpful assistant. Think it over and answer the user question correctly.<|im_end|>"
                    },  **{**{
                        "max_length":1024*16,
                        "top_p":0.95,
                        "temperature":0.01,
                    },**llm_config,**{
                        "generation.early_stopping":False,
                        "generation.repetition_penalty":1.1,
                        "generation.stop_token_ids":[151643,151645]}})       
    return t

def qwen_chat_wrapper(llm:"ByzerLLM",conversations: Optional[List[Dict]] = None,llm_config={}):
    
    
    for conv in conversations:
        if conv["role"] == "system":
            if "<|im_start|>" not in conv["content"]:
                conv["content"] = "<|im_start|>system\n" + conv["content"] + "<|im_end|>"
            
    t = llm.chat_oai(conversations=conversations,role_mapping={
                    "user_role":"<|im_start|>user\n",
                    "assistant_role": "<|im_end|>\n<|im_start|>assistant\n",
                    "system_msg":"<|im_start|>system\nYou are a helpful assistant. Think it over and answer the user question correctly.<|im_end|>"
                    },  **{**{
                        "max_length":1024*16,
                        "top_p":0.95,
                        "temperature":0.01,
                    },**llm_config,**{
                        "generation.early_stopping":False,
                        "generation.repetition_penalty":1.1,
                        "generation.stop_token_ids":[151643,151645]}})       
    v = t[0].output
    if "<|im_end|>" in v:
        v = v.split("<|im_end|>")[0]
    if "<|endoftext|>" in v:
        v = v.split("<|endoftext|>")[0]
    t[0].output = v    
    return t


            




            