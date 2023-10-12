from typing import Optional,List
import json

class ClusterSettings:
    def __init__(self, name:str, location:str, numNodes:int):
        self.name = name
        self.location = location 
        self.numNodes = numNodes   

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False)        

class TableSettings:
    def __init__(self, database:str, table:Optional[str], schema:str, location:Optional[str], num_shards:int):
        self.database = database
        self.table = table
        self.schema = schema
        self.location = location
        self.num_shards = num_shards

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False)    


class EnvSettings:
    def __init__(self, javaHome:str, path:str):
        self.javaHome = javaHome
        self.path = path   

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False)  


class ResourceRequirement:
    def __init__(self, name:float, resourceQuantity:float):
        self.name = name
        self.resourceQuantity = resourceQuantity   

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False)


class ResourceRequirementSettings:
    def __init__(self, resourceRequirements: List[ResourceRequirement]):
        self.resourceRequirements = resourceRequirements

    def json(self):
        return json.dumps({"resourceRequirements":[item.__dict__ for item in self.resourceRequirements]},ensure_ascii=False)           


class JVMSettings:
    def __init__(self, options:list[str]):
        self.options = options   

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False)         


class SearchQuery:
    def __init__(self, keyword:Optional[str], fields:list[str], vector:list[float], vectorField:Optional[str], limit:int=10):
        self.keyword = keyword
        self.fields = fields
        self.vector = vector
        self.vectorField = vectorField
        self.limit = limit

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False)    
        
    
