import ray
from pyjava import RayContext
from pyjava.udf import UDFMaster,UDFWorker,UDFBuilder,UDFBuildInFunc
from typing import Any, NoReturn, Callable, Dict, List
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from byzerllm import consume_model
from byzerllm.records import SearchQuery
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.records import SearchQuery
import json

def search_func(model,v):
    data = [json.loads(item) for item in v]
    results=[]
    for item in data:
        cluster_name = item["clusterName"]
        database = item["database"]
        table = item["table"]
        vector = []
        if "query.vector" in item:
            vector_str = item["query.vector"] 
            if vector_str.startswith("["):
                vector = json.loads(vector_str)
            else:
                vector = [float(i) for i in vector_str.split(",")]

        fields = []
        if "query.fields" in item:
            fields_str = item["query.fields"]
            if fields_str.startswith("["):
                fields = json.loads(fields_str)
            else:
                fields = fields_str.split(",")

        keyword = None
        if "query.keyword"  in item:
            keyword = item["query.keyword"]

        vector_field = None
        if "query.vectorField" in item:
            vector_field = item["query.vectorField"]    

        query = SearchQuery(
            keyword=keyword,
            fields=fields,
            vector=vector,
            vectorField=vector_field,
            limit=int(item.get("query.limit",10)),
        )
        docs = model.search(cluster_name,database,table,query)
        results.append(docs)
    return {"value":[json.dumps(docs,ensure_ascii=False)]}
                 
def init_retrieval_client(model_refs: List[ClientObjectRef], conf: Dict[str, str]) -> Any:
    consume_model(conf)
    byzer = ByzerRetrieval()
    byzer.launch_gateway()
    return byzer

