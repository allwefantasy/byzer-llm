from prometheus_client import CollectorRegistry, Gauge,Counter, pushadd_to_gateway
from byzerllm.utils.config import get_mlsql_config_pushgateway_address,get_mlsql_config
from typing import Union,Dict
import ray

class Metric:

    def __init__(self):
        self.registry = CollectorRegistry()
        config = get_mlsql_config()
        self.metric_enabled = False
        self.pushgateway_address = None
        if config is not None:
            self.pushgateway_address = ray.get(config.getitem.remote("spark.mlsql.pushgateway.address",None))
            self.metric_enabled = True

        self.gauges = {}
        self.counters = {}        

    def inc(self, name:str,value: Union[int, float] = 1.0, tags: Dict[str, str] = None):
        if not self.metric_enabled:
            return
        if name not in self.counters:
            self.counters[name] = Counter(name, '', registry=self.registry)                    
        self.counters[name].inc(value)   

    def push(self):
        if not self.metric_enabled:
            return
        if self.pushgateway_address is not None:
            pushadd_to_gateway(self.pushgateway_address, job='pushgateway', registry=self.registry)      
        
    

