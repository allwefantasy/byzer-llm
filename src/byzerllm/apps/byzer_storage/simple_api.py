from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.client import ByzerLLM, InferBackend, LLMRequest
from byzerllm.utils.client.byzerllm_client import Templates
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
import asyncio
import threading
import hashlib
import time
from byzerllm.utils.client.byzerllm_client import Template
from langchain.prompts import PromptTemplate


def generate_md5_hash(input_string: str) -> str:
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode("utf-8"))
    return md5_hash.hexdigest()


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


class SortOption(Enum):
    DESC = "desc"
    ASC = "asc"


class FilterBuilder:
    def __init__(self, query_builder: "QueryBuilder"):
        self.conditions = []
        self.query_builder = query_builder

    def add_condition(self, field: str, value: Any):
        self.conditions.append({"field": field, "value": value})
        return self

    def build(self):
        return self.conditions


class AndBuilder(FilterBuilder):
    def build(self):
        v = {"and": super().build()}
        self.query_builder.filters = v
        return self.query_builder


class OrBuilder(FilterBuilder):
    def build(self):
        v = {"or": super().build()}
        self.query_builder.filters = v
        return self.query_builder


class QueryBuilder:
    def __init__(self, storage: "ByzerStorage"):
        self.storage = storage
        self.keyword = None
        self.vector = []
        self.vector_field = None
        self.filters = {}
        self.fields = []
        self.limit = 10
        self.sorts = []

    def set_search_query(self, query: str, fields: Union[List[str], str]):
        self.keyword = self.storage.tokenize(query)
        if isinstance(fields, str):
            self.fields = fields.split(",")
        else:
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
            self.vector_field = fields.split(",")[0]
        else:
            self.vector_field = fields[0]
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
            print(self.filters)
            self.storage.retrieval.delete_by_filter(
                self.storage.cluster_name,
                self.storage.database,
                self.storage.table,
                self.filters,
            )
            self.storage.retrieval.commit(
                self.storage.cluster_name, self.storage.database, self.storage.table
            )
        else:
            raise ValueError("Only support delete by filter")


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
            raise ValueError(
                "At least one of vector_fields or search_fields is required."
            )
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
            raise ValueError(
                "At least one of vector_fields or search_fields is required."
            )
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


class ModelWriteBuilder:
    def __init__(self, storage: "ByzerStorage"):
        self.storage = storage
        self.memories = []
        self.options = {}
        self.cutoff_len = 1024
        self.stage = "pt"
        self.max_samples = 1000000
        self.per_device_train_batch_size = 2
        self.gradient_accumulation_steps = 4
        self.num_train_epochs = 1000.0
        self.num_gpus = 1.0

    def add_memory(self, memory: str):
        self.memories.append(memory)
        return self

    def add_memories(self, memories: List[str]):
        self.memories.extend(memories)
        return self

    def add_option(self, key: str, value: Any):
        self.options[key] = value
        return self

    def add_options(self, options: Dict[str, Any]):
        self.options.update(options)
        return self

    def set_cutoff_len(self, cutoff_len: int):
        self.cutoff_len = cutoff_len
        return self

    def set_max_samples(self, max_samples: int):
        self.max_samples = max_samples
        return self

    def set_stage(self, stage: str):
        self.stage = stage
        return self

    def set_per_device_train_batch_size(self, per_device_train_batch_size: int):
        self.per_device_train_batch_size = per_device_train_batch_size
        return self

    def set_gradient_accumulation_steps(self, gradient_accumulation_steps: int):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        return self

    def set_num_train_epochs(self, num_train_epochs: float):
        self.num_train_epochs = num_train_epochs
        return self

    def set_num_gpus(self, num_gpus: float):
        self.num_gpus = num_gpus
        return self

    def execute(self, remote: bool = True):
        config = {
            "cutoff_len": self.cutoff_len,
            "max_samples": self.max_samples,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "stage": self.stage,
            **self.options,
        }
        return self.storage.memorize(
            self.memories, remote=remote, options=config, num_gpus=self.num_gpus
        )


class ByzerStorage:
    _is_connected = False

    @classmethod
    def get_base_dir(cls, base_dir: str = None):
        home = os.path.expanduser("~")
        base_dir = base_dir or os.path.join(home, ".auto-coder")
        return base_dir

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

        byzerllm.connect_cluster(
            address=ray_address,
            code_search_path=code_search_path,
            init_options={"log_to_driver": True},
        )
        cls._is_connected = True
        return base_dir

    def __init__(
        self,
        cluster_name: Optional[str],
        database: str,
        table: str,
        base_dir: Optional[str] = None,
        ray_address: str = "auto",
        emb_model: str = "emb",
    ):
        self.base_dir = ByzerStorage.get_base_dir(base_dir)
        if not ByzerStorage._is_connected:
            ByzerStorage._connect_cluster(cluster_name, base_dir, ray_address)

        self.retrieval = ByzerRetrieval()
        self.retrieval.launch_gateway()
        self.cluster_name = cluster_name or "byzerai_store"
        self.emb_model = emb_model
        self.database = database
        self.table = table
        self.memory_manager = None
        self.llm = ByzerLLM()
        self.llm.setup_default_emb_model_name(self.emb_model)

    def query_builder(self) -> QueryBuilder:
        return QueryBuilder(self)

    def write_builder(self) -> WriteBuilder:
        return WriteBuilder(self)

    def schema_builder(self) -> SchemaBuilder:
        return SchemaBuilder(self)

    def write_model_builder(self) -> ModelWriteBuilder:
        return ModelWriteBuilder(self)

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
        search_query = SearchQuery(
            database=self.database,
            table=self.table,
            keyword=keyword,
            vector=vector or [],
            vectorField=vector_field,
            filters=filters or {},
            fields=fields or [],
            sorts=sorts,
            limit=limit,
        )
        if search_query.filters and not search_query.keyword and not vector_field:
            return self.retrieval.filter(self.cluster_name, search_query)
        return self.retrieval.search(self.cluster_name, search_query)

    def _add(self, data: List[Dict[str, Any]]) -> bool:
        """
        Build index from a list of dictionaries.
        """
        return self.retrieval.build_from_dicts(
            self.cluster_name, self.database, self.table, data
        )

    def quick_memory(self, memories: List[str], options: Dict[str, Any] = {}):
        if not self.retrieval.check_table_exists(
            self.cluster_name, self.database, self.table
        ):
            _ = (
                self.schema_builder()
                .add_field("_id", DataType.STRING)
                .add_field("name", DataType.STRING)
                .add_field("content", DataType.STRING, [FieldOption.ANALYZE])
                .add_field("raw_content", DataType.STRING, [FieldOption.NO_INDEX])
                .add_array_field("summary", DataType.FLOAT)
                .add_field("created_time", DataType.LONG, [FieldOption.SORT])
                .execute()
            )
        data = [
            {
                "_id": f"{self.database}_{self.table}_{generate_md5_hash(item)}",
                "name": "short_memory",
                "content": item,
                "raw_content": item,
                "summary": item,
                "created_time": int(time.time()),
            }
            for item in memories
        ]

        self.write_builder().add_items(
            data, vector_fields=["summary"], search_fields=["content"]
        ).execute()

    def memorize(
        self,
        memories: List[str],
        remote: bool = True,
        options: Dict[str, Any] = {},
        num_gpus: float = 1,
    ):
        if not remote:

            def run():
                from byzerllm.apps.byzer_storage.memory_model_based import MemoryManager

                memory_manager = MemoryManager(self, self.base_dir, remote=False)
                self.memory_manager = memory_manager
                name = f"{self.database}_{self.table}"
                asyncio.run(memory_manager.memorize(name, memories, options))

            task = threading.Thread(target=run)
            task.start()
            logger.info("Memorization task started.")
            return self.memory_manager
        else:
            import ray
            from byzerllm.apps.byzer_storage.memory_model_based import MemoryManager

            name = f"{self.database}_{self.table}"
            mm = (
                ray.remote(MemoryManager)
                .options(name=name, num_gpus=num_gpus, lifetime="detached")
                .remote(self, self.base_dir, remote=True)
            )
            mm.memorize.remote(f"{self.database}_{self.table}", memories, options)
            logger.info(
                f"Memorization task started. Please check ray dashboard for actor: `{name}`"
            )
            return mm

    def cancel_job(self):
        import ray

        name = f"{self.database}_{self.table}"
        ray.kill(ray.get_actor(name))

    def template(self):
        def llama3():
            def clean_func(v):
                return v

            def sys_format(t, v):
                m = PromptTemplate.from_template(t)
                return m.format(system_msg=v)

            return Template(
                role_mapping={
                    "user_role": "<|start_header_id|>user<|end_header_id|>",
                    "assistant_role": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                    "system_msg": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|>""",
                    "system_msg_func": sys_format,
                },
                generation_config={
                    "generation.repetition_penalty": 1.1,
                    "generation.stop_token_ids": [128000, 128001],
                },
                clean_func=clean_func,
            )

        return llama3()

    def remember(self, query: str):
        llm = ByzerLLM()
        llm.setup_default_model_name("long_memory")
        llm.setup_template("long_memory", self.template())
        name = f"{self.database}_{self.table}"
        loras_dir = os.path.join(self.base_dir, "storage", "loras")
        target_lora_dir = os.path.join(loras_dir, f"{name}")

        ## lora_name 和 lora_int_id 两个参数后续需要修改
        v = llm.chat_oai(
            conversations=[{"role": "user", "content": query}],
            llm_config={
                "gen.adapter_name_or_path": f"{target_lora_dir}",
                "gen.lora_name": "default",
                "gen.lora_int_id": 1,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        )
        return [v[0].output]

    def tokenize(self, s: str):
        seg_list = jieba.cut(s, cut_all=False)
        # return self.llm.apply_sql_func("select mkString(' ',parse(value)) as value",[
        # {"value":s}],url=self.byzer_engine_url)["value"]
        return " ".join(seg_list)

    def emb(self, s: str):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return self.llm.emb(self.llm.default_emb_model_name, LLMRequest(instruction=s))[0].output
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to get embedding after {max_retries} attempts: {str(e)}")
                    raise
                
                # Sleep between 1-5 seconds before retrying
                sleep_time = 1 + (retry_count * 1.5)
                logger.warning(f"Embedding API call failed (attempt {retry_count}/{max_retries}). Error: {str(e)}. Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

    def commit(self) -> bool:
        """
        Commit changes to the storage.
        """
        return self.retrieval.commit(self.cluster_name, self.database, self.table)

    def delete_by_ids(self, ids: List[Union[str, int]]):
        self.retrieval.delete_by_ids(self.cluster_name, self.database, self.table, ids)
        self.retrieval.commit(self.cluster_name, self.database, self.table)

    def truncate_table(self):        
        self.retrieval.truncate(
            self.cluster_name, self.database, self.table
        )        
        self.retrieval.commit(self.cluster_name, self.database, self.table) 

    def drop(self):
        self.retrieval.closeAndDeleteFile(self.cluster_name, self.database, self.table)
        self.retrieval.commit(self.cluster_name, self.database, self.table)
