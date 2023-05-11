from pyjava.api.mlsql import RayContext,PythonContext
from pyjava.storage import streaming_tar
from typing import Dict,Generator
from dataclasses import dataclass

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