import dataclasses
from typing import Optional

@dataclasses.dataclass
class ClusterSetting:
    name:str
    location:str

@dataclasses.dataclass
class TableSettings:
    database:str
    table:Optional[str]
    schema:str
    location:Optional[str] 
    num_shards:int

@dataclasses.dataclass
class EnvSettings: 
    javaHome:str
    path:str

@dataclasses.dataclass
class SearchQuery:
       keyword: Optional[str] = None 
       fields: list[str]
       vector: list[float]
       vectorField: Optional[str] = None
       limit: int=10
