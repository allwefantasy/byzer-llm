from datasets import load_dataset,Dataset
from typing import Any,List,Dict
from ray.util.client.common import ClientObjectRef
from pyjava.api.mlsql import RayContext
from pyjava.storage import streaming_tar

import ray

from typing import Dict,Generator
from dataclasses import dataclass
import logging

@dataclass
class BlockRow:
    start: int
    offset: int
    value: bytes

def restore_model(conf: Dict[str, str],target_dir:str):
    model_servers = RayContext.parse_servers(conf["modelServers"])
    model_binary = RayContext.collect_from(model_servers)
    streaming_tar.save_rows_as_file(model_binary,target_dir)

def load_model(target_dir:str)-> Generator[BlockRow,None,None]:
    model_binary = streaming_tar.build_rows_from_file(target_dir)
    return model_binary

def consume_model(conf: Dict[str, str]):
    model_servers = RayContext.parse_servers(conf["modelServers"])
    for item in RayContext.collect_from(model_servers):
        pass   

def common_init_model(model_refs: List[ClientObjectRef], 
                      conf: Dict[str, str],model_dir:str,is_load_from_local:bool):
    if not is_load_from_local:      
      if "standalone" in conf and conf["standalone"]=="true":
          logging.info(f"Standalone mode(normally only one UDFWorker). Try to restore the model from socket server to {model_dir}")
          restore_model(conf,model_dir)
      else:
          logging.info(f"Noraml mode(normally 1+ UDFWorker). Try to restore the model from ray object store to {model_dir}")
          streaming_tar.save_rows_as_file((ray.get(ref) for ref in model_refs),model_dir)
    else:
      logging.info(f"Load model from local path ({model_dir}), consume the model server to prevent socket server leak.")
      consume_model(conf)    
    