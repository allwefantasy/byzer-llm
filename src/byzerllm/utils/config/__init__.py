import ray

MLSQL_CONFIG = "__MLSQL_CONFIG__"

@ray.remote
class MLSQLConifg(object):
    def __init__(self,instance_name:str, json_obj):        
        self._config = json_obj
        self.instance_name = instance_name

    def getitem(self, key):
        return self._config[key]

    def setitem(self, key, value):
        self._config[key] = value   

    def delitem(self, key):
        del self._config[key]


def create_mlsql_config(name,json_obj):
    return MLSQLConifg.options(name=MLSQL_CONFIG,lifetime="detached").remote(json_obj)