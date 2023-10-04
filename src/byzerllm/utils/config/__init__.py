import ray

MLSQL_CONFIG = "__MLSQL_CONFIG__"

@ray.remote
class MLSQLConifg(object):
    def __init__(self,instance_name:str, json_obj):        
        self._config = json_obj
        self.instance_name = instance_name

    def getitem(self, key,defaulValue):
        return self._config.get(key,defaulValue)

    def setitem(self, key, value):
        self._config[key] = value   

    def delitem(self, key):
        del self._config[key]


def create_mlsql_config(name,json_obj):
    config = get_mlsql_config()
    if config is not None:
        ray.kill(config)
    return MLSQLConifg.options(name=MLSQL_CONFIG,lifetime="detached").remote(name,json_obj)

def get_mlsql_config():
    try:
        config = ray.get_actor(MLSQL_CONFIG)
    except:
        config = None    
    return config

def get_mlsql_config_item(key,defaultValue):
    config = get_mlsql_config()
    return ray.get(config.getitem.remote(key,defaultValue))
        
def get_mlsql_config_pushgateway_address():
    return get_mlsql_config_item("spark.mlsql.pushgateway.address",None)