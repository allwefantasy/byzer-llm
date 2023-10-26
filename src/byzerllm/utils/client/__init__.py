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
            ** request.extra_params.__dict__,
            ** extract_params}] 
        else: 
            v = [{
            "instruction":x,
            "embedding":True,
            ** request.extra_params.__dict__,
            ** extract_params} for x in request.instruction]    
        res = self._query(model,v) 
      
        return [LLMResponse(output=item["predict"],input=item["input"]) for item in res]
    
    def chat(self,model,request:LLMRequest,extract_params:Dict[str,Any]={})->List[str]:
        if isinstance(request.instruction,str):
            v = [{
            "instruction":request.instruction,            
            ** request.extra_params.__dict__,
            ** extract_params}] 
        else: 
            v = [{
            "instruction":request.instruction,            
            ** request.extra_params.__dict__,
            ** extract_params} for x in request.instruction]         
        res = self._query(model,v) 
        return [LLMResponse(output=item["predict"],input=item["input"]) for item in res]

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
            