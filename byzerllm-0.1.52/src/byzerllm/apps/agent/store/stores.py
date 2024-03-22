from typing import Union
from . import MessageStore
import ray
from ray.util.client.common import ClientActorHandle

class Stores:
    def __init__(self,store:Union[str,MessageStore,ClientActorHandle]):        
        if isinstance(store,str):
            try:
                self.store = ray.get_actor(store)
            except:
                print(f"Store {store} not found",flush=True)
                pass
        else:
            self.store = store

    def put(self, message):
        if isinstance(self.store,MessageStore):
            return self.store.put(message)
        else:
            return self.store.put.remote(message)

    def get(self, id):
        if isinstance(self.store,MessageStore):
            return self.store.get(id)
        else:
            return ray.get(self.store.get.remote(id))
        
    def clear(self,id):
        if isinstance(self.store,MessageStore):
            return self.store.clear(id)
        else:
            return ray.get(self.store.clear.remote(id))     