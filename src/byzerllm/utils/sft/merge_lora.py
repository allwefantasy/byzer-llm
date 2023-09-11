from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyjava.api.mlsql import DataServer
from pyjava.storage import streaming_tar as STar
import torch
from typing import Any,Any,Dict, List,Tuple,Generator
from .. import BlockRow

def merge_lora_to_base_model(data_refs:List[DataServer],
              train_params:Dict[str,str],
              conf: Dict[str, str])->Generator[BlockRow,Any,Any]:
    
    model_name_or_path = train_params.get("modelNameOrPath",train_params.get("model_name_or_path",""))
    adapter_name_or_path = train_params.get("adapterNameOrPath",train_params.get("adapter_name_or_path",""))
    save_path = train_params.get("savePath",train_params.get("save_path",""))
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    return STar.build_rows_from_file(save_path)    
