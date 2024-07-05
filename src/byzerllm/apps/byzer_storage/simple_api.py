from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.records import SearchQuery, TableSettings
from typing import List, Dict, Any, Union
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
    def __init__(self):
        self.fields = []

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

    def build(self) -> str:
        return f"st({','.join(self.fields)})"

class ByzerStorage:
    def __init__(self, cluster_name: str, database: str, table: str):
        self.retrieval = ByzerRetrieval()
        self.cluster_name = cluster_name
        self.database = database
        self.table = table

    def query_builder(self) -> QueryBuilder:
        return QueryBuilder(self)

    def write_builder(self) -> WriteBuilder:
        return WriteBuilder(self)

    def schema_builder(self) -> SchemaBuilder:
        return SchemaBuilder()

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

    def initialize(self, schema: str, num_shards: int = 1):
        """
        Initialize the storage by creating the table if it doesn't exist.
        """
        if not self.retrieval.check_table_exists(self.cluster_name, self.database, self.table):
            table_settings = TableSettings(
                database=self.database,
                table=self.table,
                schema=schema,
                location=None,  # Let the system decide the location
                num_shards=num_shards
            )
            self.retrieval.create_table(self.cluster_name, table_settings)

    def commit(self) -> bool:
        """
        Commit changes to the storage.
        """
        return self.retrieval.commit(self.cluster_name, self.database, self.table)
