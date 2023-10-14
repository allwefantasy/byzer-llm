from fastapi import FastAPI
import ray
from ray import serve
from typing import List,Dict,Any
import json
from . import ByzerRetrieval,ClusterSettings, EnvSettings, JVMSettings, TableSettings,SearchQuery,ResourceRequirementSettings


app = FastAPI()

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class SimpleRest:
    
    def __init__(self):                        
        self.retrieval = ByzerRetrieval() 
        self.retrieval.launch_gateway()    
        

    @app.post("/cluster/create")
    def cluster(self,   cluster_settings:str,                       
                        env_settings:str, 
                        jvm_settings:str,
                        resource_requirement_settings:str) -> str:
        cluster_settings_obj = ClusterSettings(**json.loads(cluster_settings))
        env_settings_obj = EnvSettings(**json.loads(env_settings))
        jvm_settings_obj = JVMSettings(**json.loads(jvm_settings))
        resource_requirement_settings_obj = JVMSettings(**json.loads(resource_requirement_settings))
        return self.retrieval.start_cluster(cluster_settings_obj,
                                            env_settings_obj,
                                            jvm_settings_obj,
                                            resource_requirement_settings_obj);
    
    @app.get("/cluster")                                        
    def cluster_info(self,name:str) -> str:
        return json.dumps(self.retrieval.cluster_info(name),ensure_ascii=False)
    
    @app.post("/cluster/restore")                                        
    def restore_from_cluster_info(self,cluster_info:str) -> str:
        return json.dumps({
            "status":self.retrieval.restore_from_cluster_info(json.loads(cluster_info))
        },ensure_ascii=False)
    
    @app.post("/table/create")                                        
    def create_table(self,cluster_name:str,table_settings:str) -> str:
        table_settings_obj = TableSettings(**json.loads(table_settings))
        return json.dumps({
            "status":self.retrieval.create_table(cluster_name,table_settings_obj)
        },ensure_ascii=False)
    
    @app.post("/table/data") 
    def build(self, cluster_name:str, database:str, table:str, data:str)-> bool:
        data_list = json.loads(data)
        data_refs = []        
        for item in data_list:
            itemref = ray.put(json.dumps(item,ensure_ascii=False))
            data_refs.append(itemref)

        return json.dumps({
            "status":self.retrieval.build(cluster_name,database,table,data_refs)
        },ensure_ascii=False)
    
    @app.post("/table/commit")
    def commit(self,cluster_name:str, database:str, table:str)-> bool:
        return json.dumps({
            "status":self.retrieval.commit(cluster_name,database,table)
        },ensure_ascii=False)
    
    @app.post("/table/search")
    def search(self,cluster_name:str, database:str, table:str, query:str)-> str:
        query_obj = SearchQuery(**json.loads(query))
        return json.dumps(self.retrieval.search(cluster_name,database,table,query_obj),ensure_ascii=False)
        
    
    @app.post("/table/close")
    def close(self,cluster_name:str,database:str,table:str):
        return json.dumps({
            "status":self.retrieval.close(cluster_name,database,table)
        },ensure_ascii=False)
    
    @app.post("/table/truncate")
    def close(self,cluster_name:str,database:str,table:str):
        return json.dumps({
            "status":self.retrieval.truncate(cluster_name,database,table)
        },ensure_ascii=False)
    
    @app.post("/table/close_and_delete_file")
    def closeAndDeleteFile(self,cluster_name:str,database:str,table:str):
        return json.dumps({
            "status":self.retrieval.closeAndDeleteFile(cluster_name,database,table)
        },ensure_ascii=False)


def deploy():
    serve.run(SimpleRest.bind(),route_prefix="/retrievel")
