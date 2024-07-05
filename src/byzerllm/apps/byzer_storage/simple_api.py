from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.records import SearchQuery, TableSettings, ClusterSettings, EnvSettings, JVMSettings, ResourceRequirementSettings, ResourceRequirement
from typing import List, Dict, Any, Union, Optional
from enum import Enum, auto

class DataType(Enum):
    STRING = "string"
    LONG = "long"
    INT = "int"
    FLOAT = "float"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    ARRAY = "array"
    MAP = "map"
    STRUCT = "st"

class FieldOption(Enum):
    ANALYZE = "analyze"
    SORT = "sort"
    NO_INDEX = "no_index"

class QueryBuilder:
    def __init__(self, storage: 'ByzerStorage'):
        self.storage = storage
        self.keyword = None
        self.vector = None
        self.vector_field = None
        self.filters = {}
        self.fields = []
        self.limit = 10

    def set_keyword(self, keyword: str):
        self.keyword = keyword
        return self

    def set_vector(self, vector: List[float], vector_field: str):
        self.vector = vector
        self.vector_field = vector_field
        return self

    def add_filter(self, field: str, value: Any):
        self.filters[field] = value
        return self

    def set_fields(self, fields: List[str]):
        self.fields = fields
        return self

    def set_limit(self, limit: int):
        self.limit = limit
        return self

    def execute(self) -> List[Dict[str, Any]]:
        return self.storage.query(
            keyword=self.keyword,
            vector=self.vector,
            vector_field=self.vector_field,
            filters=self.filters,
            fields=self.fields,
            limit=self.limit
        )

class WriteBuilder:
    def __init__(self, storage: 'ByzerStorage'):
        self.storage = storage
        self.data = []

    def add_item(self, item: Dict[str, Any]):
        self.data.append(item)
        return self

    def add_items(self, items: List[Dict[str, Any]]):
        self.data.extend(items)
        return self

    def execute(self) -> bool:
        return self.storage.add(self.data)

class SchemaBuilder:
    def __init__(self, storage: 'ByzerStorage'):
        self.fields = []
        self.storage = storage
        self.num_shards = 1
        self.location = None

    def add_field(self, name: str, data_type: DataType, options: List[FieldOption] = None):
        field = f"field({name},{data_type.value}"
        if options:
            field += f",{','.join([opt.value for opt in options])}"
        field += ")"
        self.fields.append(field)
        return self

    def add_array_field(self, name: str, element_type: DataType, options: List[FieldOption] = None):
        field = f"field({name},array({element_type.value})"
        if options:
            field += f",{','.join([opt.value for opt in options])}"
        field += ")"
        self.fields.append(field)
        return self

    def add_map_field(self, name: str, key_type: DataType, value_type: DataType, options: List[FieldOption] = None):
        field = f"field({name},map({key_type.value},{value_type.value})"
        if options:
            field += f",{','.join([opt.value for opt in options])}"
        field += ")"
        self.fields.append(field)
        return self

    def add_struct_field(self, name: str, struct_builder: 'SchemaBuilder', options: List[FieldOption] = None):
        field = f"field({name},{struct_builder.build()}"
        if options:
            field += f",{','.join([opt.value for opt in options])}"
        field += ")"
        self.fields.append(field)
        return self

    def set_num_shards(self, num_shards: int):
        self.num_shards = num_shards
        return self

    def set_location(self, location: str):
        self.location = location
        return self

    def build(self) -> str:
        return f"st({','.join(self.fields)})"

    def execute(self) -> bool:
        schema = self.build()
        table_settings = TableSettings(
            database=self.storage.database,
            table=self.storage.table,
            schema=schema,
            location=self.location,
            num_shards=self.num_shards
        )
        return self.storage.retrieval.create_table(self.storage.cluster_name, table_settings)

class SimpleClusterBuilder:
    def __init__(self, storage: 'ByzerStorage'):
        self.storage = storage
        self.name = None
        self.location = None
        self.num_nodes = 1
        self.node_memory = "2g"
        self.node_cpu = 1
        self.java_home = None
        self.path = None

    def set_name(self, name: str):
        self.name = name
        return self

    def set_location(self, location: str):
        self.location = location
        return self

    def set_num_nodes(self, num_nodes: int):
        self.num_nodes = num_nodes
        return self

    def set_node_memory(self, node_memory: str):
        self.node_memory = node_memory
        return self

    def set_node_cpu(self, node_cpu: int):
        self.node_cpu = node_cpu
        return self

    def set_java_home(self, java_home: str):
        self.java_home = java_home
        return self

    def set_path(self, path: str):
        self.path = path
        return self

    def build(self) -> bool:
        if not all([self.name, self.location, self.java_home, self.path]):
            raise ValueError("Name, location, Java home, and path are required.")

        cluster_settings = ClusterSettings(self.name, self.location, self.num_nodes)
        env_settings = EnvSettings(self.java_home, self.path)
        jvm_settings = JVMSettings([f"-Xmx{self.node_memory}", "-XX:+UseZGC"])
        resource_settings = ResourceRequirementSettings([ResourceRequirement("CPU", self.node_cpu)])

        return self.storage.retrieval.start_cluster(
            cluster_settings,
            env_settings,
            jvm_settings,
            resource_settings
        )

class ByzerStorage:
    def __init__(self, cluster_name: str, database: str, table: str):
        self.retrieval = ByzerRetrieval()
        self.retrieval.launch_gateway()
        self.cluster_name = cluster_name
        self.database = database
        self.table = table

    def query_builder(self) -> QueryBuilder:
        return QueryBuilder(self)

    def write_builder(self) -> WriteBuilder:
        return WriteBuilder(self)

    def schema_builder(self) -> SchemaBuilder:
        return SchemaBuilder(self)

    def cluster_builder(self) -> SimpleClusterBuilder:
        return SimpleClusterBuilder(self)

    def query(self, keyword: str = None, vector: List[float] = None, 
               vector_field: str = None, filters: Dict[str, Any] = None, 
               fields: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Unified search method supporting both keyword and vector search.
        """
        search_query = SearchQuery(
            database=self.database,
            table=self.table,
            keyword=keyword,
            vector=vector or [],
            vectorField=vector_field,
            filters=filters or {},
            fields=fields or [],
            limit=limit
        )
        return self.retrieval.search(self.cluster_name, search_query)

    def add(self, data: List[Dict[str, Any]]) -> bool:
        """
        Build index from a list of dictionaries.
        """
        return self.retrieval.build_from_dicts(self.cluster_name, self.database, self.table, data)

    def commit(self) -> bool:
        """
        Commit changes to the storage.
        """
        return self.retrieval.commit(self.cluster_name, self.database, self.table)

    def start_cluster(self, name: str, location: str, java_home: str, path: str, 
                      num_nodes: int = 1, node_memory: str = "2g", node_cpu: int = 1) -> bool:
        """
        Simplified method to start a cluster.
        """
        return self.cluster_builder() \
            .set_name(name) \
            .set_location(location) \
            .set_java_home(java_home) \
            .set_path(path) \
            .set_num_nodes(num_nodes) \
            .set_node_memory(node_memory) \
            .set_node_cpu(node_cpu) \
            .build()
