import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import time

from pydantic import BaseModel
from byzerllm import SimpleByzerLLM, ByzerLLM
from byzerllm.records import (
    SearchQuery,
    TableSettings,
    ClusterSettings,
    EnvSettings,
    JVMSettings,
    ResourceRequirementSettings,
    ResourceRequirement,
)
from loguru import logger
try:
    import pyarrow as pa
except ImportError:
    pa = None
    logger.error("pyarrow is not installed, please install it by `pip install pyarrow`")

try:
    import pyarrow.flight as flight
except ImportError:
    flight = None
    logger.error("pyarrow.flight is not included in pyarrow, please install it by `pip install pyarrow[flight]`")    


import jieba
class SearchQuery(BaseModel):
    database: str
    table: str
    keyword: Optional[str] = None
    vector: List[float] = []
    vectorField: Optional[str] = None
    filters: Dict[str, Any] = {}
    fields: List[str] = []
    sorts: Optional[List[Dict[str, str]]] = None
    limit: int = 10
    
    def json(self):
        return json.dumps(self.dict(), ensure_ascii=False)

# 集成 RetrievalClient 类
class RetrievalClient:
    def __init__(self, host: str = "localhost", port: int = 33333):
        """Initialize the retrieval client with connection to Arrow Flight server.
        
        Args:
            host: The hostname of the Arrow Flight server
            port: The port of the Arrow Flight server
        """
        self.client = flight.FlightClient(f"grpc://{host}:{port}")
    
    def create_table(self, database: str, table: str, schema: str, 
                     location: str, num_shards: int) -> bool:
        """Create a new table in the retrieval system.
        
        Args:
            database: The database name
            table: The table name
            schema: The schema definition in string format
            location: The location to store the table data
            num_shards: Number of shards for the table
        
        Returns:
            True if successful, False otherwise
        """
        # Create a record batch with the table settings
        batch_data = [
            pa.array([database]),
            pa.array([table]),
            pa.array([schema]),
            pa.array([location]),
            pa.array([num_shards], type=pa.int32())
        ]
        
        batch = pa.RecordBatch.from_arrays(
            batch_data,
            names=['database', 'table', 'schema', 'location', 'numShards']
        )
        
        # Convert to IPC message
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        
        # Send action to server
        action = flight.Action("CreateTable", sink.getvalue().to_pybytes())
        results = list(self.client.do_action(action))
        
        if results and len(results) > 0:
            return results[0].body.to_pybytes().decode('utf-8') == "true"
        return False
    
    def check_table_exists(self, cluster_name: str, database: str, table: str) -> bool:
        """Check if the table exists in the retrieval system.
        
        Args:
            cluster_name: The name of the cluster
            database: The database name
            table: The table name
        
        Returns:
            True if the table exists, False otherwise
        """
        # Create record batch with cluster_name
        batch_data = [
            pa.array([cluster_name])
        ]
        
        batch = pa.RecordBatch.from_arrays(
            batch_data,
            names=['cluster_name']
        )
        
        # Convert to IPC message
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        
        # Send ClusterInfo action to server
        action = flight.Action("ClusterInfo", sink.getvalue().to_pybytes())
        results = list(self.client.do_action(action))
                        
        cluster_info_json = results[0].body.to_pybytes().decode('utf-8')
        cluster_info = json.loads(cluster_info_json)
        
        target_table_settings = None
        for table_settings_dict in cluster_info["tableSettingsList"]:
            table_settings = TableSettings(**table_settings_dict)
            if table_settings.database == database and table_settings.table == table:
                target_table_settings = table_settings
                break
        if target_table_settings:
            return True
        return False
            
            
        
    def build_from_local(self, cluster_name: str, database: str, table: str, data: List[Dict[str, Any]]) -> bool:
        """Build the table from local data.
        
        Args:
            database: The database name
            table: The table name
            data: List of dictionaries containing the data
        
        Returns:
            True if successful, False otherwise
        """
        # Convert data to JSON strings
        json_data = [json.dumps(item) for item in data]
        
        # Create a record batch with the data
        batch_data = [
            pa.array([database]),
            pa.array([table]),
            pa.array([json_data])
        ]
        
        batch = pa.RecordBatch.from_arrays(
            batch_data,
            names=['database', 'table', 'data']
        )
        
        # Convert to IPC message
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        
        # Send action to server
        action = flight.Action("BuildFromLocal", sink.getvalue().to_pybytes())
        results = list(self.client.do_action(action))        
        
        if results and len(results) > 0:
            return results[0].body.to_pybytes().decode('utf-8') == "true"
        return False
    
    def search(self, database: str, table: str, 
               keyword: Optional[str] = None,
               vector: Optional[List[float]] = None,
               vector_field: Optional[str] = None,
               filters: Optional[List[Dict[str, Any]]] = None,
               sorts: Optional[List[Dict[str, str]]] = None,
               fields: Optional[List[str]] = None,
               limit: int = 10) -> List[Dict[str, Any]]:
        """使用给定的查询参数搜索表。
        
        参数:
            database: 数据库名称
            table: 表名称
            keyword: 可选的文本搜索关键词
            vector: 可选的向量搜索值
            vector_field: 向量搜索的字段名
            filters: 可选的过滤条件
            sorts: 可选的排序条件
            fields: 可选的返回字段列表
            limit: 返回结果的最大数量
        
        返回:
            匹配文档的列表
        """
        # 构建搜索查询        
        search_query = SearchQuery(
            database=database,
            table=table,
            keyword=keyword,
            vector=vector or [],
            vectorField=vector_field,
            filters=filters[0] if filters else {},  # 取第一个过滤条件
            fields=fields or [],
            sorts=sorts,
            limit=limit,
        )
        # 将查询转换为JSON
        query_json = f"[{search_query.json()}]"
        
        # 创建包含查询的记录批次
        batch_data = [
            pa.array([database]),
            pa.array([table]),
            pa.array([query_json])
        ]
        
        batch = pa.RecordBatch.from_arrays(
            batch_data,
            names=['database', 'table', 'query']
        )
        
        # 转换为IPC消息
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        
        # 向服务器发送操作
        action = flight.Action("Search", sink.getvalue().to_pybytes())
        results = list(self.client.do_action(action))
        
        if results and len(results) > 0:
            result_json = results[0].body.to_pybytes().decode('utf-8')
            return json.loads(result_json)
        return []
    
    def commit(self, database: str, table: str) -> bool:
        """Commit changes to the table.
        
        Args:
            database: The database name
            table: The table name
        
        Returns:
            True if successful, False otherwise
        """
        # Create a record batch with the database and table
        batch_data = [
            pa.array([database]),
            pa.array([table])
        ]
        
        batch = pa.RecordBatch.from_arrays(
            batch_data,
            names=['database', 'table']
        )
        
        # Convert to IPC message
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        
        # Send action to server
        action = flight.Action("Commit", sink.getvalue().to_pybytes())
        results = list(self.client.do_action(action))
        
        if results and len(results) > 0:
            return results[0].body.to_pybytes().decode('utf-8') == "true"
        return False
    
    def delete_by_filter(self, database: str, table: str, condition: str) -> bool:
        """Delete documents matching the condition.
        
        Args:
            database: The database name
            table: The table name
            condition: Filter condition in JSON format
        
        Returns:
            True if successful, False otherwise
        """
        # Create a record batch with the filter condition
        batch_data = [
            pa.array([database]),
            pa.array([table]),
            pa.array([condition])
        ]
        
        batch = pa.RecordBatch.from_arrays(
            batch_data,
            names=['database', 'table', 'condition']
        )
        
        # Convert to IPC message
        sink = pa.BufferOutputStream()
        writer = pa.RecordBatchStreamWriter(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        
        # Send action to server
        action = flight.Action("DeleteByFilter", sink.getvalue().to_pybytes())
        results = list(self.client.do_action(action))
        
        if results and len(results) > 0:
            return results[0].body.to_pybytes().decode('utf-8') == "true"
        return False
    
    def shutdown(self) -> bool:
        """Shutdown the retrieval server.
        
        Returns:
            True if successful, False otherwise
        """
        action = flight.Action("Shutdown", b"")
        results = list(self.client.do_action(action))
        
        if results and len(results) > 0:
            return results[0].body.to_pybytes().decode('utf-8') == "true"
        return False

def generate_md5_hash(input_string: str) -> str:
    """
    Generate MD5 hash for a string
    """
    import hashlib
    return hashlib.md5(input_string.encode()).hexdigest()

class DataType(Enum):
    STRING = "string"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    ARRAY = "array"
    MAP = "map"
    STRUCT = "struct"


class FieldOption(Enum):
    ANALYZE = "analyze"
    SORT = "sort"
    NO_INDEX = "no_index"


class SortOption(Enum):
    DESC = "desc"
    ASC = "asc"


class FilterBuilder:
    def __init__(self, query_builder: "QueryBuilder"):
        self.query_builder = query_builder
        self.conditions = []

    def add_condition(self, field: str, value: Any):
        self.conditions.append({"field": field, "value": value})
        return self

    def build(self):
        pass


class AndBuilder(FilterBuilder):
    def build(self):
        self.query_builder.filters = {"and": self.conditions}
        return self.query_builder


class OrBuilder(FilterBuilder):
    def build(self):
        self.query_builder.filters = {"or": self.conditions}
        return self.query_builder


class QueryBuilder:
    def __init__(self, storage: "LocalByzerStorage"):
        self.storage = storage
        self.filters = {}
        self.keyword = None
        self.vector = None
        self.vector_field = None
        self.fields = None
        self.sorts = []
        self.limit = 10

    def set_search_query(self, query: str, fields: Union[List[str], str]):
        self.keyword = query
        if isinstance(fields, str):
            fields = [fields]
        self.fields = fields
        return self

    def set_vector_query(
        self, query: Union[List[float], str], fields: Union[List[str], str]
    ):
        if isinstance(query, str):
            self.vector = self.storage.emb(query)
        else:
            self.vector = query

        if isinstance(fields, str):
            self.vector_field = fields
        else:
            self.vector_field = fields[0] if fields else None

        return self

    def and_filter(self) -> AndBuilder:
        return AndBuilder(self)

    def or_filter(self) -> OrBuilder:
        return OrBuilder(self)

    def sort(self, field: str, order: SortOption = SortOption.DESC):
        self.sorts.append({field: order.value})
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
            sorts=self.sorts,
            limit=self.limit,
        )

    def delete(self):
        if self.filters and not self.keyword and not self.vector_field:
            filter_json = json.dumps(self.filters)
            self.storage.retrieval.delete_by_filter(
                self.storage.database,
                self.storage.table,
                filter_json
            )
            self.storage.retrieval.commit(
                self.storage.database, self.storage.table
            )
        else:
            raise ValueError("Only support delete by filter")


class WriteBuilder:
    def __init__(self, storage: "LocalByzerStorage"):
        self.storage = storage
        self.items = []

    def add_item(
        self,
        item: Dict[str, Any],
        vector_fields: List[str] = [],
        search_fields: List[str] = [],
    ):        
        if not vector_fields and not search_fields:
            raise ValueError(
                "At least one of vector_fields or search_fields is required."
            )
        for field in vector_fields:
            item[field] = self.storage.emb(item[field])
        
        for field in search_fields:
            item[field] = self.storage.tokenize(item[field])
        
        self.items.append(item)
        return self

    def add_items(
        self,
        items: List[Dict[str, Any]],
        vector_fields: List[str] = [],
        search_fields: List[str] = [],
    ):
        for item in items:
            self.add_item(item, vector_fields, search_fields)
        return self

    def execute(self) -> bool:
        return self.storage._add(self.items)


class SchemaBuilder:
    def __init__(self, storage: "LocalByzerStorage"):
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
            table_settings.database, table_settings.table, table_settings.schema, table_settings.location, table_settings.num_shards
        )


class LocalByzerStorage:
    def __init__(
        self,
        cluster_name: str,
        database: str,
        table: str,
        host: str = "localhost",
        port: int = 33333,
        emb_llm: Union[ByzerLLM, SimpleByzerLLM] = None
    ):
        self.retrieval = RetrievalClient(host=host, port=port)
        self.cluster_name = cluster_name
        self.database = database
        self.table = table
        self.llm = emb_llm

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
        sorts: List[Dict[str, str]] = [],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Unified search method supporting both keyword and vector search.
        """
        # Convert filters dict to list format expected by RetrievalClient
        filters_list = None
        if filters:
            filters_list = [filters]

        # Convert sorts to the format expected by RetrievalClient
        sorts_list = None
        if sorts:
            sorts_list = sorts
        
        return self.retrieval.search(
            database=self.database,
            table=self.table,
            keyword=keyword,
            vector=vector,
            vector_field=vector_field,
            filters=filters_list,
            sorts=sorts_list,
            fields=fields,
            limit=limit
        )

    def _add(self, data: List[Dict[str, Any]]) -> bool:
        """
        Build index from a list of dictionaries.
        """
        return self.retrieval.build_from_local(
            self.cluster_name, self.database, self.table, data
        )

    def commit(self) -> bool:
        """
        Commit changes to the table.
        """
        return self.retrieval.commit(self.database, self.table)

    def delete_by_ids(self, ids: List[Union[str, int]]):
        """
        Delete documents by ID.
        """
        filter_condition = {"or": [{"field": "id", "value": id} for id in ids]}
        filter_json = json.dumps(filter_condition)
        return self.retrieval.delete_by_filter(
            self.database, self.table, filter_json
        )

    def truncate_table(self):
        """
        Delete all documents in the table.
        """
        filter_json = json.dumps({"matchAll": True})
        return self.retrieval.delete_by_filter(
            self.database, self.table, filter_json
        )

    def drop(self):
        """
        Drop the table.
        """
        # There might not be a direct equivalent in RetrievalClient
        # This is a placeholder - in a real implementation, you might need to
        # implement this functionality in RetrievalClient first
        return True

    def emb(self, s: str) -> List[float]:
        """
        Embed a string using an embedding model.
        This is a mock implementation - in a real scenario, you'd use
        an actual embedding model or service.
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return self.llm.emb_query(s)[0].output
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to get embedding after {max_retries} attempts: {str(e)}")
                    raise
                
                # Sleep between 1-5 seconds before retrying
                sleep_time = 1 + (retry_count * 1.5)
                logger.warning(f"Embedding API call failed (attempt {retry_count}/{max_retries}). Error: {str(e)}. Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

    def tokenize(self, s: str):
        seg_list = jieba.cut(s, cut_all=False)
        # return self.llm.apply_sql_func("select mkString(' ',parse(value)) as value",[
        # {"value":s}],url=self.byzer_engine_url)["value"]
        return " ".join(seg_list)
