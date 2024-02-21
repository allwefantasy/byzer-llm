
import ray 
from ray.types import ObjectRef
from byzerllm.records import ClusterSettings, EnvSettings, JVMSettings, TableSettings,SearchQuery,ResourceRequirementSettings,ResourceRequirement
from typing import List,Dict,Any,Optional,Union
import byzerllm.utils.object_store_ref_util as ref_utils
import json

class ClusterBuilder:

    def __init__(self,br:'ByzerRetrieval') -> None:
        self.name = None
        self.location = None
        self.numNodes = 1
        self.nodeMemory = "2g"
        self.nodeCPU = 1
        self.enableZGC = True
        self.javaHome = None
        self.path = None

        self.cluster_settings = None
        self.env_settings = None
        self.jvm_settings = None
        self.resource_requirement_settings = None

        self.custom_resources = {}

        self.br = br

    def set_name(self,name:str):
        self.name = name
        return self
    
    def set_location(self,location:str):
        self.location = location
        return self
    
    def set_num_nodes(self,numNodes:int):
        self.numNodes = numNodes
        return self
    
    def set_node_memory(self,nodeMemory:str):
        self.nodeMemory = nodeMemory
        return self
    
    def set_custom_resource(self,k:str,v:float):
        self.custom_resources[k] = v
        return self

    
    def set_node_cpu(self,nodeCPU:int):
        self.nodeCPU = nodeCPU
        return self
    
    def set_enable_zgc(self):
        self.enableZGC = True
        return self
    
    def set_java_home(self,javaHome:str):
        self.javaHome = javaHome
        return self
    
    def set_path(self,path:str):
        self.path = path
        return self
        

    def build(self):     

        if self.name is None:
            raise Exception("name is required")
        
        if self.location is None:
            raise Exception("location is required")
        
        self.cluster_settings = ClusterSettings(self.name,self.location,self.numNodes)

        if self.javaHome is None:
            raise Exception("javaHome is required")
        
        if self.path is None:
            raise Exception("path is required")        

        self.env_settings = EnvSettings(javaHome=self.javaHome,path=self.path)

        jvmOptions = []
        resourceOptions = []
        if self.enableZGC:
            jvmOptions.append("-XX:+UseZGC")

        if self.nodeMemory:
            jvmOptions.append(f"-Xmx{self.nodeMemory}")

        if self.nodeCPU:
           resourceOptions.append(ResourceRequirement("CPU",self.nodeCPU))

        if self.custom_resources:
            for k,v in self.custom_resources.items():
                resourceOptions.append(ResourceRequirement(k,v))   

        self.jvm_settings = JVMSettings(jvmOptions)
        self.resource_requirement_settings = ResourceRequirementSettings(resourceOptions)        
    
    def start_cluster(self)-> bool:     
        self.build()
        return self.br.start_cluster(self.cluster_settings,self.env_settings,self.jvm_settings,self.resource_requirement_settings)


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

    def cluster_builder(self) -> ClusterBuilder:
        br = self
        return ClusterBuilder(br)

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
    
    def is_cluster_exists(self,name:str) -> bool:
        try:
            ray.get_actor(name)
            return True
        except ValueError:
            return False
    
    def get_table_settings(self,cluster_name:str, database:str, table:str) -> Optional[TableSettings]:               
        cluster_info = self.cluster_info(cluster_name)
        target_table_settings = None
        for table_settings_dict in cluster_info["tableSettingsList"]:
            table_settings = TableSettings(**table_settings_dict)
            if table_settings.database == database and table_settings.table == table:
                target_table_settings = table_settings
                break        
        return target_table_settings
    
    def check_table_exists(self,cluster_name:str, database:str, table:str) -> bool:
        return self.get_table_settings(cluster_name,database,table) is not None
        
    
    def restore_from_cluster_info(self,cluster_info:Dict[str,Any]) -> bool:        
        return ray.get(self.retrieval_gateway.restoreFromClusterInfo.remote(json.dumps(cluster_info,ensure_ascii=False)))

    def create_table(self,cluster_name:str, tableSettings:TableSettings)-> bool:
        
        if self.check_table_exists(cluster_name,tableSettings.database,tableSettings.table):
            raise Exception(f"Table {tableSettings.database}.{tableSettings.table} already exists in cluster {cluster_name}")

        cluster = self.cluster(cluster_name)
        return ray.get(cluster.createTable.remote(tableSettings.json()))     

    def build(self, cluster_name:str, database:str, table:str, object_refs:List[ObjectRef[str]])-> bool:
        
        if not self.check_table_exists(cluster_name,database,table):
            raise Exception(f"Table {database}.{table} not exists in cluster {cluster_name}")
        
        cluster = self.cluster(cluster_name)
        
        data_ids = ref_utils.get_object_ids(object_refs)
        locations = ref_utils.get_locations(object_refs)
        return ray.get(cluster.buildFromRayObjectStore.remote(database,table,data_ids,locations))
    
    def build_from_dicts(self, cluster_name:str, database:str, table:str, data:List[Dict[str,Any]])-> bool:
        data_refs = []

        for item in data:
            itemref = ray.put(json.dumps(item ,ensure_ascii=False))
            data_refs.append(itemref)
        
        return self.build(cluster_name,database,table,data_refs)

    def delete_by_ids(self,cluster_name:str, database:str, table:str,ids:List[Any])-> bool:

        if not self.check_table_exists(cluster_name,database,table):
            raise Exception(f"Table {database}.{table} not exists in cluster {cluster_name}")

        cluster = self.cluster(cluster_name)
        return ray.get(cluster.deleteByIds.remote(database,table,json.dumps(ids,ensure_ascii=False)))
    
    def get_tables(self,cluster_name:str) -> List[TableSettings]:
        cluster_info = self.cluster_info(cluster_name)
        target_table_settings = []
        for table_settings_dict in cluster_info["tableSettingsList"]:
            target_table_settings.append(TableSettings(**table_settings_dict))
        return target_table_settings
    
    def get_databases(self,cluster_name:str) -> List[str]:
        table_settings_list = self.get_tables(cluster_name)
        return [x.database for x in table_settings_list]
        
    
    def shutdown_cluster(self,cluster_name:str)->bool:
        if not self.launched:
            raise Exception("Please launch gateway first")
                
        v = ray.get(self.retrieval_gateway.shutdownCluster.remote(cluster_name))
        if cluster_name in self.clusters:
            del self.clusters[cluster_name]        
        return v
            

    def commit(self,cluster_name:str, database:str, table:str)-> bool:
        
        if not self.check_table_exists(cluster_name,database,table):
            raise Exception(f"Table {database}.{table} not exists in cluster {cluster_name}")
        
        cluster = self.cluster(cluster_name)
        return ray.get(cluster.commit.remote(database,table))
    
    def truncate(self,cluster_name:str, database:str, table:str)-> bool:

        if not self.check_table_exists(cluster_name,database,table):
            raise Exception(f"Table {database}.{table} not exists in cluster {cluster_name}")

        cluster = self.cluster(cluster_name)
        return ray.get(cluster.truncate.remote(database,table))
    
    def close(self,cluster_name:str, database:str, table:str)-> bool:

        if not self.check_table_exists(cluster_name,database,table):
            raise Exception(f"Table {database}.{table} not exists in cluster {cluster_name}")

        cluster = self.cluster(cluster_name)
        return ray.get(cluster.close.remote(database,table))
    
    def closeAndDeleteFile(self,cluster_name:str, database:str, table:str)-> bool:

        if not self.check_table_exists(cluster_name,database,table):
            raise Exception(f"Table {database}.{table} not exists in cluster {cluster_name}")

        cluster = self.cluster(cluster_name)
        return ray.get(cluster.closeAndDeleteFile.remote(database,table))
    
    def search_keyword(self,cluster_name:str, 
                       database:str, 
                       table:str, 
                       filters: Dict[str,Any],
                       keyword:str, 
                       fields:List[str], 
                       limit:int=10) -> List[Dict[str,Any]]:                

        search = SearchQuery(database=database,table=table,filters=filters,keyword=keyword,fields=fields,vector=[],vectorField=None,limit=limit)
        cluster = self.cluster(cluster_name)
        v = cluster.search.remote(f"[{search.json()}]")
        return json.loads(ray.get(v))
    
    def search_vector(self,cluster_name:str, 
                       database:str, 
                       table:str, 
                       filters: Dict[str,Any],
                       vector:List[float], 
                       vector_field:str,                        
                       limit:int=10) -> List[Dict[str,Any]]:
                
        search = SearchQuery(database=database,table=table,filters=filters,keyword=None,fields=[],vector=vector,vectorField=vector_field,limit=limit)
        cluster = self.cluster(cluster_name)
        v = cluster.search.remote(f"[{search.json()}]")
        return json.loads(ray.get(v))
    
    def search(self,cluster_name:str,search_query: Union[List[SearchQuery],SearchQuery]) -> List[Dict[str,Any]]:        
        cluster = self.cluster(cluster_name)
        if isinstance(search_query,SearchQuery):
            search_query = [search_query]

        if not search_query:
            raise Exception("search_query is empty")
            
        v = cluster.search.remote(f"[{','.join([x.json() for x in search_query])}]")
        return json.loads(ray.get(v))  
    
    def filter(self,cluster_name:str,search_query: Union[List[SearchQuery],SearchQuery]) -> List[Dict[str,Any]]:        
        cluster = self.cluster(cluster_name)
        if isinstance(search_query,SearchQuery):
            search_query = [search_query]
        v = cluster.filter.remote(f"[{','.join([x.json() for x in search_query])}]")
        return json.loads(ray.get(v))

    def delete_by_filter(self,cluster_name:str, database:str, table:str,filter:Dict[str,Any])-> bool:
        cluster = self.cluster(cluster_name)
        return ray.get(cluster.deleteByFilter.remote(database,table,json.dumps(filter,ensure_ascii=False)))    


    
    
    

        


      