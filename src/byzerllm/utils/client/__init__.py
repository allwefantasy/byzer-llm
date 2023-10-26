from pyjava.udf import UDFMaster
from pyjava import RayContext
from typing import Dict,Any,List,Optional
from pyjava.udf import UDFBuilder
import ray

class ByzerLLM:
    def __init__(self,url:Optional[str],**kwargs):
        self.url = url       
        if url is not None:            
            v = globals()
            self.context = v["context"]
            self.ray_context = RayContext.connect(v, self.url, **kwargs)

    def chat(self, model:str, input_value:List[Dict[str,Any]]):
        udf_master = ray.get_actor(model)
        try:
            [index, worker] = ray.get(udf_master.get.remote())
            res = ray.get(worker.apply.remote(input_value))
            return res
        except Exception as inst:
            raise inst
        finally:
            ray.get(udf_master.give_back.remote(index))        
            

        
