from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.records import (
    SearchQuery,
    TableSettings,
    ClusterSettings,
    EnvSettings,
    JVMSettings,
    ResourceRequirementSettings,
    ResourceRequirement,
)
from typing import List, Dict, Any, Union, Optional
from enum import Enum, auto
import os
import byzerllm
import jieba


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
    def __init__(self, storage: "ByzerStorage"):
        self.storage = storage
        self.keyword = None
        self.vector = None
        self.vector_field = None
        self.filters = {}
        self.fields = []
        self.limit = 10

    def set_keyword(self, keyword: str, fields: Optional[List[str]] = None):
        self.keyword = keyword
        if fields:
            self.fields = fields
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
        return self.storage._query(
            keyword=self.keyword,
            vector=self.vector,
            vector_field=self.vector_field,
            filters=self.filters,
            fields=self.fields,
            limit=self.limit,
        )


class WriteBuilder:
    def __init__(self, storage: "ByzerStorage"):
        self.storage = storage
        self.data = []

    def add_item(self, item: Dict[str, Any]):
        self.data.append(item)
        return self

    def add_items(self, items: List[Dict[str, Any]]):
        self.data.extend(items)
        return self

    def execute(self) -> bool:
        return self.storage._add(self.data)


class SchemaBuilder:
    def __init__(self, storage: "ByzerStorage"):
        self.fields = []
        self.storage = storage
        self.num_shards = 1
        self.location = ""

    def add_field(
        self, name: str, data_type: DataType, options: List[FieldOption] = None
    ):
        field = f"field({name},{data_type.value}"
        if options:
            field += f",{','.join([opt.value for opt in options])}"
        field += ")"
        self.fields.append(field)
        return self

    def add_array_field(
        self, name: str, element_type: DataType, options: List[FieldOption] = None
    ):
        field = f"field({name},array({element_type.value})"
        if options:
            field += f",{','.join([opt.value for opt in options])}"
        field += ")"
        self.fields.append(field)
        return self

    def add_map_field(
        self,
        name: str,
        key_type: DataType,
        value_type: DataType,
        options: List[FieldOption] = None,
    ):
        field = f"field({name},map({key_type.value},{value_type.value})"
        if options:
            field += f",{','.join([opt.value for opt in options])}"
        field += ")"
        self.fields.append(field)
        return self

    def add_struct_field(
        self,
        name: str,
        struct_builder: "SchemaBuilder",
        options: List[FieldOption] = None,
    ):
        field = f"field({name},{struct_builder.build()}"
        if options:
            field += f",{','.join([opt.value for opt in options])}"
        field += ")"
        self.fields.append(field)
        return self

    def build(self) -> str:
        return f"st({','.join(self.fields)})"

    def execute(self) -> bool:
        if self.storage.retrieval.check_table_exists(
            self.storage.cluster_name, self.storage.database, self.storage.table
        ):
            return False

        schema = self.build()
        table_settings = TableSettings(
            database=self.storage.database,
            table=self.storage.table,
            schema=schema,
            location=self.location,
            num_shards=self.num_shards,
        )
        return self.storage.retrieval.create_table(
            self.storage.cluster_name, table_settings
        )


class ByzerStorage:
    _is_connected = False

    @classmethod
    def _connect_cluster(
        cls,
        cluster_name: Optional[str],
        base_dir: Optional[str] = None,
        ray_address: str = "auto",
    ):
        if cls._is_connected:
            return True

        version = "0.1.11"
        cluster = cluster_name
        home = os.path.expanduser("~")
        base_dir = base_dir or os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(
            base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}"
        )
        cluster_json = os.path.join(base_dir, "storage", "data", f"{cluster}.json")

        if not os.path.exists(cluster_json) or not os.path.exists(libs_dir):
            print("No instance found.")
            return False

        code_search_path = [libs_dir]

        byzerllm.connect_cluster(address=ray_address, code_search_path=code_search_path)
        cls._is_connected = True
        return True

    def __init__(
        self,
        cluster_name: Optional[str],
        database: str,
        table: str,
        base_dir: Optional[str] = None,
        ray_address: str = "auto",
    ):
        if not ByzerStorage._is_connected:
            ByzerStorage._connect_cluster(cluster_name, base_dir, ray_address)
        self.retrieval = ByzerRetrieval()
        self.retrieval.launch_gateway()
        self.cluster_name = cluster_name or "byzerai_store"
        self.database = database
        self.table = table

    def query_builder(self) -> QueryBuilder:
        return QueryBuilder(self)

    def write_builder(self) -> WriteBuilder:
        return WriteBuilder(self)

    def schema_builder(self) -> SchemaBuilder:
        return SchemaBuilder(self)

    def _query(
        self,
        keyword: str = None,
        vector: List[float] = None,
        vector_field: str = None,
        filters: Dict[str, Any] = None,
        fields: List[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
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
            limit=limit,
        )
        return self.retrieval.search(self.cluster_name, search_query)

    def _add(self, data: List[Dict[str, Any]]) -> bool:
        """
        Build index from a list of dictionaries.
        """
        return self.retrieval.build_from_dicts(
            self.cluster_name, self.database, self.table, data
        )

    def setokenize(self, s: str):
        seg_list = jieba.cut(s, cut_all=False)
        # return self.llm.apply_sql_func("select mkString(' ',parse(value)) as value",[
        # {"value":s}],url=self.byzer_engine_url)["value"]
        return " ".join(seg_list)

    def commit(self) -> bool:
        """
        Commit changes to the storage.
        """
        return self.retrieval.commit(self.cluster_name, self.database, self.table)
