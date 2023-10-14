
import ray 
from ray.types import ObjectRef
from byzerllm.records import ClusterSettings, EnvSettings, JVMSettings, TableSettings,SearchQuery,ResourceRequirementSettings
from typing import List,Dict,Any
import byzerllm.utils.object_store_ref_util as ref_utils
import json


class ByzerRetrieval:
    
    def __init__(self):
        self.launched = False
        self.retrieval_gateway = None
        self.clusters = {}

    def launch_gateway(self)-> ray.actor.ActorHandle:
        
        try:
           self.retrieval_gateway = ray.get_actor("RetrievalGateway")
        except Exception:
            pass   

        if self.retrieval_gateway:
            self.launched = True
            return self.retrieval_gateway

        if self.launched:
            return ray.get_actor("RetrievalGateway")
        
        retrieval_gateway_launcher_clzz = ray.cross_language.java_actor_class("tech.mlsql.retrieval.RetrievalGatewayLauncher")
        retrieval_gateway_launcher = retrieval_gateway_launcher_clzz.remote()
        ray.get(retrieval_gateway_launcher.launch.remote()) 
        retrieval_gateway = ray.get_actor("RetrievalGateway")
        self.launched = True
        self.retrieval_gateway = retrieval_gateway
        return retrieval_gateway  

    def gateway(slef) -> ray.actor.ActorHandle:
        return ray.get_actor("RetrievalGateway")
                 

    def start_cluster(self, cluster_settings:ClusterSettings,                       
                      env_settings:EnvSettings, 
                      jvm_settings:JVMSettings,
                      resource_requirement_settings:ResourceRequirementSettings = ResourceRequirementSettings([])) -> bool:                      
        if not self.launched:
            raise Exception("Please launch gateway first")
        
        if cluster_settings.name in self.clusters:
            raise Exception(f"Cluster {cluster_settings.name} already exists")
        
        try:
            ray.get_actor(cluster_settings.name)
            raise Exception(f"Cluster {cluster_settings.name} already exists")   
        except ValueError:
            pass
        
        obj_ref1 = self.retrieval_gateway.buildCluster.remote(
                    cluster_settings.json(),                    
                    env_settings.json(),
                    jvm_settings.json(),
                    resource_requirement_settings.json()
                    )
        
        return ray.get(obj_ref1) 
    
    def cluster(self,name:str) -> ray.actor.ActorHandle:
        if not self.launched:
            raise Exception("Please launch gateway first")
        
        if name in self.clusters:
            return self.clusters[name]
        
        cluster_ref = self.retrieval_gateway.getCluster.remote(name)
        # master_ref.buildFromRayObjectStore.remote("db1","table1",data_refs)
        cluster = ray.get(cluster_ref)
        self.clusters[name] = cluster
        return cluster
    
    def cluster_info(self,name:str) -> Dict[str,Any]:
        cluster = self.cluster(name)
        return json.loads(ray.get(cluster.clusterInfo.remote()))
    
    def restore_from_cluster_info(self,cluster_info:Dict[str,Any]) -> bool:        
        return ray.get(self.retrieval_gateway.restoreFromClusterInfo.remote(json.dumps(cluster_info,ensure_ascii=False)))

    def create_table(self,cluster_name:str, tableSettings:TableSettings)-> bool:
        cluster = self.cluster(cluster_name)
        return ray.get(cluster.createTable.remote(tableSettings.json()))    

    def build(self, cluster_name:str, database:str, table:str, object_refs:List[ObjectRef[str]])-> bool:
        cluster = self.cluster(cluster_name)
        
        data_ids = ref_utils.get_object_ids(object_refs)
        locations = ref_utils.get_locations(object_refs)
        return ray.get(cluster.buildFromRayObjectStore.remote(database,table,data_ids,locations))

    def commit(self,cluster_name:str, database:str, table:str)-> bool:
        cluster = self.cluster(cluster_name)
        return ray.get(cluster.commit.remote(database,table))
    
    def truncate(self,cluster_name:str, database:str, table:str)-> bool:
        cluster = self.cluster(cluster_name)
        return ray.get(cluster.truncate.remote(database,table))
    
    def close(self,cluster_name:str, database:str, table:str)-> bool:
        cluster = self.cluster(cluster_name)
        return ray.get(cluster.close.remote(database,table))
    
    def closeAndDeleteFile(self,cluster_name:str, database:str, table:str)-> bool:
        cluster = self.cluster(cluster_name)
        return ray.get(cluster.closeAndDeleteFile.remote(database,table))
    
    def search_keyword(self,cluster_name:str, 
                       database:str, 
                       table:str, 
                       keyword:str, 
                       fields:List[str], 
                       limit:int=10) -> List[Dict[str,Any]]:
        search = SearchQuery(keyword=keyword,fields=fields,vector=[],vectorField=None,limit=limit)
        cluster = self.cluster(cluster_name)
        v = cluster.search.remote(database,table,search.json())
        return json.loads(ray.get(v))
    
    def search_vector(self,cluster_name:str, 
                       database:str, 
                       table:str, 
                       vector:List[float], 
                       vector_field:str,                        
                       limit:int=10) -> List[Dict[str,Any]]:
        search = SearchQuery(keyword=None,fields=[],vector=vector,vectorField=vector_field,limit=limit)
        cluster = self.cluster(cluster_name)
        v = cluster.search.remote(database,table,search.json())
        return json.loads(ray.get(v))
    
    def search(self,cluster_name:str, 
                       database:str, 
                       table:str, 
                       search_query: SearchQuery) -> List[Dict[str,Any]]:        
        cluster = self.cluster(cluster_name)
        v = cluster.search.remote(database,table,search_query.json())
        return json.loads(ray.get(v))  


    
    
    

        


      