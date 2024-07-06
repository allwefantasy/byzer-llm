from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.client import ByzerLLM, InferBackend, LLMRequest
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
import jieba
from loguru import logger


class DataType(Enum):
    STRING = "string"
    LONG = "long"    
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


class FilterBuilder:
    def __init__(self):
        self.conditions = []

    def add_condition(self, field: str, value: Any):
        self.conditions.append({"field": field, "value": value})
        return self

    def build(self):
        return self.conditions

class AndBuilder(FilterBuilder):
    def build(self):
        return {"and": super().build()}

class OrBuilder(FilterBuilder):
    def build(self):
        return {"or": super().build()}

class QueryBuilder:
    def __init__(self, storage: "ByzerStorage"):
        self.storage = storage
        self.keyword = None
        self.vector = None
        self.vector_field = None
        self.filters = None
        self.fields = []
        self.limit = 10

    def set_search_query(self, query: str, fields: Optional[List[str]] = None):
        self.keyword = self.storage.tokenize(query)
        if fields:
            self.fields = fields
        return self

    def set_vector_query(self, query: Union[List[float], str], fields: Optional[List[str]]):
        if isinstance(query, str):
            self.vector = self.storage.emb(query)
        else:
            self.vector = query
        self.vector_field = fields[0]
        return self

    def and_filter(self) -> AndBuilder:
        return AndBuilder()

    def or_filter(self) -> OrBuilder:
        return OrBuilder()    

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

    def add_item(
        self,
        item: Dict[str, Any],
        vector_fields: List[str] = [],
        search_fields: List[str] = [],
    ):
        if not vector_fields and not search_fields:
            raise ValueError("At least one of vector_fields or search_fields is required.")
        for field in vector_fields:
            item[field] = self.storage.emb(item[field])
        for field in search_fields:
            item[field] = self.storage.tokenize(item[field])
        self.data.append(item)
        return self

    def add_items(
        self,
        items: List[Dict[str, Any]],
        vector_fields: List[str] = [],
        search_fields: List[str] = [],
    ):
        if not vector_fields and not search_fields:
            raise ValueError("At least one of vector_fields or search_fields is required.")
        for item in items:
            for field in vector_fields:
                item[field] = self.storage.emb(item[field])
            for field in search_fields:
                item[field] = self.storage.tokenize(item[field])
            self.data.append(item)
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
        schema = self.build()  
        logger.info(schema)
        if self.storage.retrieval.check_table_exists(
            self.storage.cluster_name, self.storage.database, self.storage.table
        ):            
            return False
              
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

        import byzerllm

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
        emb_model: str = "emb",
    ):
        if not ByzerStorage._is_connected:
            ByzerStorage._connect_cluster(cluster_name, base_dir, ray_address)
        self.retrieval = ByzerRetrieval()
        self.retrieval.launch_gateway()
        self.cluster_name = cluster_name or "byzerai_store"
        self.emb_model = emb_model
        self.database = database
        self.table = table

        self.llm = ByzerLLM()
        self.llm.setup_default_emb_model_name(self.emb_model)

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

    def tokenize(self, s: str):
        seg_list = jieba.cut(s, cut_all=False)
        # return self.llm.apply_sql_func("select mkString(' ',parse(value)) as value",[
        # {"value":s}],url=self.byzer_engine_url)["value"]
        return " ".join(seg_list)

    def emb(self, s: str):
        return self.llm.emb(self.llm.default_emb_model_name, LLMRequest(instruction=s))[
            0
        ].output

    def commit(self) -> bool:
        """
        Commit changes to the storage.
        """
        return self.retrieval.commit(self.cluster_name, self.database, self.table)
