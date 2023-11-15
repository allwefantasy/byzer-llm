from typing import Optional,List,Dict,Any,Union
import json

class ClusterSettings:
    def __init__(self, name:str, location:str, numNodes:int):
        self.name = name
        self.location = location 
        self.numNodes = numNodes   

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False)        

    @staticmethod 
    def from_json(json_str:str):
        return ClusterSettings(**json.loads(json_str))       

class TableSettings:
    def __init__(self, database:str, table:Optional[str], schema:str, location:Optional[str], num_shards:int,status:str="open"):
        self.database = database
        self.table = table
        self.schema = schema
        self.location = location
        self.num_shards = num_shards
        self.status = status

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False)

    @staticmethod 
    def from_json(json_str:str):
        return TableSettings(**json.loads(json_str))    


class EnvSettings:
    def __init__(self, javaHome:str, path:str):
        self.javaHome = javaHome
        self.path = path   

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False) 

    @staticmethod
    def from_json(json_str:str):
        return EnvSettings(**json.loads(json_str)) 


class ResourceRequirement:
    def __init__(self, name:str, resourceQuantity:float):
        self.name = name
        self.resourceQuantity = resourceQuantity   

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False)
    
    @staticmethod
    def from_json(json_str:str):
        return ResourceRequirement(**json.loads(json_str))
    


class ResourceRequirementSettings:
    def __init__(self, resourceRequirements: List[ResourceRequirement]):
        self.resourceRequirements = resourceRequirements

    def json(self):
        return json.dumps({"resourceRequirements":[item.__dict__ for item in self.resourceRequirements]},ensure_ascii=False)           
    
    @staticmethod
    def from_json(json_str:str):
        s = json.loads(json_str)
        return ResourceRequirementSettings([ResourceRequirement(**s["resourceRequirements"]) ])


class JVMSettings:
    def __init__(self, options:list[str]):
        self.options = options   

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False) 

    @staticmethod
    def from_json(json_str:str):
        return JVMSettings(**json.loads(json_str))        


class SearchQuery:
    '''
    filters: List[Dict[str,Any]] = {"and":[{"field":"name","value":"张三"}]}
    filters: List[Dict[str,Any]] = {"or":[{"field":"name","value":"张三"},{"field":"name","value":"李四"}]}
    filters: List[Dict[str,Any]] = {"or":[{"field":"name","value":"张三"},{"and":[{"field":"name","value":"李四"},{"field":"age","min":10,"max":20}]}]}}]}    
    '''
    def __init__(self,database:str,
                 table:str,                  
                 keyword:Optional[str], fields:list[str], 
                 vector:list[float], vectorField:Optional[str], filters:Dict[str,Any]={},
                 sorts: List[Dict[str,str]]=[],limit:int=10):
        self.database = database
        self.table = table
        self.filters = filters
        self.sorts = sorts
        self.keyword = keyword
        self.fields = fields
        self.vector = vector
        self.vectorField = vectorField
        self.limit = limit

    def json(self):
        return json.dumps(self.__dict__,ensure_ascii=False) 

    @staticmethod
    def from_json(json_str:str):
        return SearchQuery(**json.loads(json_str)) 


# class EqualFilter:
#     def __init__(self, field:str, value:Any):
#         self.field = field
#         self.value = value

#     def json(self):
#         return json.dumps(self.__dict__,ensure_ascii=False) 

#     @staticmethod
#     def from_json(json_str:str):
#         return EqualFilter(**json.loads(json_str))

# class RangeFilter:
#     def __init__(self, field:str, min:Any, max:Any):
#         self.field = field
#         self.min = min
#         self.max = max

#     def json(self):
#         return json.dumps(self.__dict__,ensure_ascii=False) 

#     @staticmethod
#     def from_json(json_str:str):
#         return RangeFilter(**json.loads(json_str))

# class OrRelation:
#     def __init__(self, filters:Union[EqualFilter,RangeFilter],parent:Union["OrRelation","AndRelation","QueryBuilder"]):
#         self.filters = filters

#     def json(self):
#         return json.dumps(self.__dict__,ensure_ascii=False) 

#     @staticmethod
#     def from_json(json_str:str):
#         return OrRelation(**json.loads(json_str)) 
    
#     def and_relation(self):
#         return AndRelation(self)        
    
#     def equal_filter(self,field:str,value:Any):
#         self.filters.append(EqualFilter(field,value))
#         return self
    
#     def range_filter(self,field:str,min:Any,max:Any):
#         self.filters.append(RangeFilter(field,min,max))
#         return self
    
#     def end(self):
#         return self.parent
    
#     def or_relation(self):
#         return OrRelation(self)

# class AndRelation:
#     def __init__(self, filters:Union[EqualFilter,RangeFilter],parent:Union[OrRelation,"AndRelation","QueryBuilder"]):
#         self.filters = filters
#         self.parent = parent        

#     def json(self):
#         return json.dumps(self.__dict__,ensure_ascii=False) 

#     @staticmethod
#     def from_json(json_str:str):
#         return AndRelation(**json.loads(json_str))
    
#     def equal_filter(self,field:str,value:Any):
#         self.filters.append(EqualFilter(field,value))
#         return self
    
#     def range_filter(self,field:str,min:Any,max:Any):
#         self.filters.append(RangeFilter(field,min,max))
#         return self
    
#     def end(self):
#         return self.parent
    
#     def or_relation(self):
#         return OrRelation(self)
            

# class FilterBuilder:
#     def __init__(self,query_builder:"QueryBuilder"):
#         self._filter = []
#         self.query_builder = query_builder

#     def end(self):
#         return self.query_builder
    
#     def or_relation(self):
#         return OrRelation(self)
    


# class QueryBuilder:
#     def __init__(self):
#         self._database = ""
#         self._table = ""
#         self._filter = []
#         self._keyword = ""
#         self._fields = []
#         self._vector = []
#         self._vectorField = ""
#         self._limit = 10
#         self.filter_builder = FilterBuilder(self)

#     def database(self,database:str):
#         self._database = database
#         return self

#     def table(self,table:str):
#         self._table = table
#         return self

#     def filter(self):
#         return self.filter_builder


      

        
    
