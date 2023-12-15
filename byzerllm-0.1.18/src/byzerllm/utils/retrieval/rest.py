from fastapi import Body, FastAPI
import ray
from ray import serve
from pydantic import BaseModel,Field
from typing import List,Dict,Any,Annotated,Optional
import json
from byzerllm.records import (ClusterSettings, 
                                      EnvSettings, 
                                      JVMSettings, 
                                      TableSettings,
                                      SearchQuery,
                                      ResourceRequirement,
                                      ResourceRequirementSettings)
from byzerllm.utils.retrieval import ByzerRetrieval

class ClusterSettingsParam(BaseModel):
    name:str
    location:str
    numNodes:int
    
    def cluster_settings(self):
        return ClusterSettings(**self.dict())

class EnvSettingsParam(BaseModel):
    javaHome:str
    path:str

    def env_settings(self):
        return EnvSettings(**self.dict())

class JVMSettingsParam(BaseModel):
    options:list[str]

    def jvm_settings(self):
        return JVMSettings(**self.dict())

class ResourceRequirementParam(BaseModel):
    name:str
    resourceQuantity:float

    def resource_requirement(self):
        return ResourceRequirement(**self.dict())

class ResourceRequirementSettingsParam(BaseModel):
    resourceRequirements: List[ResourceRequirementParam]

    def resource_requirement_settings(self):
        return ResourceRequirementSettings([item.resource_requirement() for item in self.resourceRequirements])

class TableSettingsParam(BaseModel):
    database:str
    table:str
    my_schema: str = Field(alias='schema')    
    location:str
    num_shards:int

    def table_settings(self):
        v = self.dict()        
        return TableSettings(**v)
    
    def dict(self):
        t = self.__dict__
        t["schema"]=t["my_schema"]
        del t["my_schema"]
        return t

class SearchQueryParam(BaseModel):
    keyword:Optional[str]
    fields:list[str]
    vector:list[float]
    vectorField:Optional[str]
    limit:int  

    def search_query(self):
        return SearchQuery(**self.dict())  
    
    def json(self):
        return json.dumps(self.dict(),ensure_ascii=False)
     
app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class SimpleRest:
    
    def __init__(self):                        
        self.retrieval = ByzerRetrieval() 
        self.retrieval.launch_gateway()    
        

    @app.post("/cluster/create")
    def cluster(self,   cluster_settings:ClusterSettingsParam,                       
                        env_settings:EnvSettingsParam, 
                        jvm_settings:JVMSettingsParam,
                        resource_requirement_settings:ResourceRequirementSettingsParam):        
        return {
            "status":self.retrieval.start_cluster(cluster_settings.cluster_settings(),
                                            env_settings.env_settings(),
                                            jvm_settings.jvm_settings(),
                                            resource_requirement_settings.resource_requirement_settings())
        }
    
    @app.get("/cluster/get/{name}")                                        
    def cluster_info(self,name:str):
        return self.retrieval.cluster_info(name)
    
    @app.post("/cluster/restore")                                        
    def restore_from_cluster_info(self,cluster_info:str) :
        return {
            "status":self.retrieval.restore_from_cluster_info(json.loads(cluster_info))
        }
    
    @app.post("/table/create/{cluster_name}")                                        
    def create_table(self,cluster_name:str,table_settings:TableSettingsParam):        
        return {
            "status":self.retrieval.create_table(cluster_name,table_settings.table_settings())
        }
    
    @app.post("/table/data") 
    def build(self, cluster_name: Annotated[str, Body()], database:Annotated[str, Body()], 
              table:Annotated[str, Body()], data:Annotated[List[Dict[str,Any]], Body()]):        
        data_refs = []        
        for item in data:
            itemref = ray.put(json.dumps(item,ensure_ascii=False))
            data_refs.append(itemref)

        return {
            "status":self.retrieval.build(cluster_name,database,table,data_refs)
        }
    
    @app.post("/table/commit")
    def commit(self,cluster_name: Annotated[str, Body()], database: Annotated[str, Body()], table: Annotated[str, Body()]):
        return {
            "status":self.retrieval.commit(cluster_name,database,table)
        }
    
    @app.post("/table/search")
    def search(self,cluster_name:Annotated[str, Body()], 
               database:Annotated[str, Body()], 
               table:Annotated[str, Body()], 
               query:SearchQueryParam):        
        return self.retrieval.search(cluster_name,database,table,query.search_query())
        
    
    @app.post("/table/close")
    def close(self,cluster_name:Annotated[str, Body()],database:Annotated[str, Body()],table:Annotated[str, Body()]):
        return {
            "status":self.retrieval.close(cluster_name,database,table)
        }
    
    @app.post("/table/truncate")
    def close(self,cluster_name:Annotated[str, Body()],database:Annotated[str, Body()],table:Annotated[str, Body()]):
        return {
            "status":self.retrieval.truncate(cluster_name,database,table)
        }
    
    @app.post("/table/close_and_delete_file")
    def closeAndDeleteFile(self,cluster_name:Annotated[str, Body()],database:Annotated[str, Body()],table:Annotated[str, Body()]):
        return {
            "status":self.retrieval.closeAndDeleteFile(cluster_name,database,table)
        }


def deploy_retrieval_rest_server(**kargs):
    # route_prefix="/retrievel",host="0.0.0.0",
    new_kargs = {**kargs}
    if "route_prefix" not in kargs:
        new_kargs["route_prefix"] = "/retrieval"
    if "host" not in kargs:
        new_kargs["host"] = "127.0.0.1"    
    serve.run(SimpleRest.bind(), **new_kargs)
