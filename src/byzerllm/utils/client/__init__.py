from pyjava.udf import UDFMaster
from pyjava import PythonContext,RayContext
from typing import Dict,Any,List,Optional,Union,Tuple,Callable
from pyjava.udf import UDFBuilder
import ray
import sys
import traceback
import io
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import json
import dataclasses
import importlib  
from . import code_utils

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
    max_length: int = 1024
    top_p: float = 0.7
    temperature: float = 0.9
    extra_params: LLMRequestExtra = LLMRequestExtra()

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
        return self
    
    def setup_gpus_per_worker(self,num_gpus:int)->'ByzerLLM':
        self.sys_conf["num_gpus"] = num_gpus
        return self

    def setup_num_workers(self,num_workers:int)->'ByzerLLM':
        self.sys_conf["masterMaxConcurrency"] = num_workers
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
            
    def _generate_ins(self,ins:str,request:LLMRequest):
         if request.extra_params.user_role:
            return f'{request.extra_params.system_msg}\n\n{request.extra_params.user_role}:{ins}\n{request.extra_params.assistant_role}:'
         return ins

    def chat(self,model,request:Union[LLMRequest,str],extract_params:Dict[str,Any]={})->List[LLMResponse]:
        if not model and not self.default_model_name:
            raise Exception("model name is required")
        
        if not model:
            model = self.default_model_name
        
        if isinstance(request,str): 
            request = LLMRequest(instruction=request)

        if isinstance(request.instruction,str):
            v = [{
            "instruction":self._generate_ins(request.instruction,request),
            "max_length":request.max_length,
            "top_p":request.top_p,
            "temperature":request.temperature,            
            ** request.extra_params.__dict__,
            ** extract_params}] 
        else: 
            v = [{
            "instruction":self._generate_ins(x,request), 
            "max_length":request.max_length,
            "top_p":request.top_p,
            "temperature":request.temperature,           
            ** request.extra_params.__dict__,
            ** extract_params} for x in request.instruction]         
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
    def __init__(self) -> None:
        pass        

    def execute(self,code)->Tuple[int, str, str]:
        return code_utils.execute_code(
                code = code,
                timeout=30*60,
                filename=None,
                work_dir=None,
                use_docker=False,
                lang="python"        
                ) 
    
    def exec_capture_output(code: str) -> Tuple[int,Any]:
        buffer = io.StringIO()
        sys.stdout = buffer
        sys.stderr = buffer

        try:
            exec(code)
        except Exception:
            return 1,traceback.format_exc()

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return 0,buffer.getvalue()

    def eval_code(self, code: str,target_names:List[str]=[]) -> Tuple[int,Any]:        
        try:
            variables = {}
            exec(code,variables)
            response = {}
            for name in target_names:
                if name in variables:
                    response[name] = variables[name]
                
            return 0,response
        except Exception as e:
            return 1,traceback.format_exc()

class ByzerLLMCoder:
    def __init__(self,llm:ByzerLLM,num_gpus=0, num_cpus=1) -> None:
        self.llm = llm
        self.sandbox = None
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus

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
        if isinstance(prompt,str):
            prompt = LLMRequest(instruction=prompt)
        
        response = self.llm.chat(None,request=prompt,extract_params=config)
        return code_utils.extract_code(response[0].output, pattern), -1 

    def improve_function(self,file_name, func_name, objective, **config):
        """Improve the function to achieve the objective."""        
        # read the entire file into a str
        with open(file_name, "r") as f:
            file_string = f.read()
        new_prompt = f'''Improve the function '{func_name}' to achieve the objective '{objective}'.
The current implementation of the function is as follows:
{file_string}'''
        response = self.llm.chat(None, request=LLMRequest(instruction=new_prompt,**config))            
        return response[0].output, -1

    def default_check_eval_repsonse(self,response:Dict[str,Any],target_names:List[str]=[])->Tuple[bool,str]:
        missing_variables = []
        for name in target_names:
            if name not in response:
                missing_variables.append(name)
        if missing_variables:
            return False,f"the response missing the variables: {missing_variables}"
        return True,""        
        
    
    def try_execute_code_until_resolved(self,code:str,target_names:List[str]=[], max_try_times:int=3)->Tuple[int, str, str]:
        status,response = self.eval_code(code,target_names)        
        max_try_times = 3        
        for i in range(max_try_times):
            if status != 0:       
                improve_response,_ = self.improve_code(code=code,objective="The code throws exception like this: {}.\n Try to fix this problem.\n".format(response))            
                lang,code = code_utils.extract_code(improve_response)[0]
                print(f"Try {i} times. The code execution failed,  the error message is: {response}. improved the code:\n{code}")                
                status,response = self.eval_code(code,target_names)                                
            else:
                if not target_names:
                    break

                success,msg = self.default_check_eval_repsonse(response,target_names)
                if success:
                    break    
                else:
                    improve_response,_ = self.improve_code(code=code,objective=f"After execute the code, {msg}.\n Try to fix this problem.\n")
                    lang,code = code_utils.extract_code(improve_response)[0]
                    print(f"Try {i} times. The code execution failed,  the error message is: {msg}. improved the code:\n{code}")                
                    status,response = self.eval_code(code,target_names)            

        return status,response        
    
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
        new_prompt = f'''Analyze the code in the following files and return a list of suggestions for improvement{followup}, to achieve the objective of '{objective}'.
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
    
    def eval_code(self, code,target_names:List[str]=[])->Tuple[int, str, str]:
        if self.sandbox is None:
            self.sandbox = ray.remote(CodeSandbox).options(
                name="CodeSandbox",                
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus
            ).remote()

        if target_names:
            status,response = ray.get(self.sandbox.eval_code.remote(code,target_names))
        else:
            status,response = ray.get(self.sandbox.exec_capture_output.remote(code))

        return status,response
            




            