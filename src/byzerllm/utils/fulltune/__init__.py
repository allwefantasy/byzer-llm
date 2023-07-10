
import os
import ray
from datetime import datetime
import uuid
import json
from typing import Dict, Any,List,Generator
from ray.util.client.common import ClientObjectRef
from pyjava.storage import streaming_tar as STar
from pyjava.api.mlsql import DataServer
from byzerllm import BlockRow
from  .deepspeed_trainner import distribute_train,TrainParameters
from pyjava.udf.store import transfer_to_ob
from byzerllm import consume_model


def sfft_train(data_refs:List[DataServer],train_params:Dict[str,str],sys_conf: Dict[str, str])->Generator[BlockRow,Any,Any]:    
    localPathPrefix = train_params.get("localPathPrefix","/tmp/byzerllm")
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    rd = f"sft-{formatted_time}-{str(uuid.uuid4())}"

    sfft_name = train_params["name"] if "name" in train_params else f"sfft-{sys_conf['OWNER']}"
    
    model_dir = os.path.join(localPathPrefix,rd,"pretrained_model")
    
    model_refs = []

    if "localModelDir" in train_params:
        model_dir = train_params["localModelDir"]
        consume_model(sys_conf)
    else:    
        transfer_to_ob(sfft_name,sys_conf,model_refs)    
           
    output_dir = os.path.join(localPathPrefix,rd,"finetune_model")
    data_dir = os.path.join(localPathPrefix,rd,"finetune_data")
    config_dir = os.path.join(localPathPrefix,rd,"config_dir")    

    print(f'''
name: {sfft_name}
model_dir: {model_dir}
output_dir: {output_dir}
data_dir: {data_dir}
config_dir: {config_dir}
          ''')
    
    train_params_sfft = {}
    
    for k,v in train_params.items():
        if k.startswith("sfft."):
            # sft.float.num_train_epochs
            tpe = k.split(".")[1]
            new_k = k.split(".")[2]
            new_v = v
            if tpe == "float":
              new_v = float(v)
            elif tpe == "int":
                new_v = int(v)
            elif tpe == "bool":
                new_v = v == "true"
            elif tpe == "str":
                new_v = v
            elif tpe == "list":
                new_v = json.loads(v)
            elif tpe == "dict":
                new_v = json.loads(v)            
            train_params_sfft[new_k] = new_v

    new_model_refs = distribute_train(TrainParameters(
        name=sfft_name,
        data_dir=data_dir,
        config_dir=config_dir,
        tokenizer_path=model_dir,
        model_dir=model_dir,
        checkpoint_saving_path=output_dir,
        model_refs = model_refs,
        ** train_params_sfft       
    ),data_refs)

    return STar.build_rows_from_file((ray.get(item) for item in new_model_refs))