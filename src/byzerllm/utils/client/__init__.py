from pyjava.udf import UDFMaster
from pyjava import RayContext
from typing import Dict,Any,List,Optional,Union
from pyjava.udf import UDFBuilder
import ray
import json
import dataclasses

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

      
class ByzerLLM:
    def __init__(self,url:Optional[str]=None,**kwargs):
        self.url = url       
        if url is not None:            
            v = globals()
            self.context = v["context"]
            self.ray_context = RayContext.connect(v, self.url, **kwargs)

    def emb(self, model, request:LLMRequest ,extract_params:Dict[str,Any]={})->List[List[float]]:
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
            return f'{request.extra_params.user_role}:{ins}\n{request.extra_params.assistant_role}:'
         return ins

    def chat(self,model,request:LLMRequest,extract_params:Dict[str,Any]={})->List[LLMResponse]:


        if isinstance(request.instruction,str):
            v = [{
            "instruction":self._generate_ins(request.instruction),
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
            