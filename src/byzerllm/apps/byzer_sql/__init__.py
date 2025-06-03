from typing import Dict,Any
from dataclasses import asdict
from byzerllm.utils.client import ByzerLLM
import json
import ray
import byzerllm
import os
from loguru import logger
from typing import Optional
from os.path import expanduser
from byzerllm import ByzerLLM,ByzerRetrieval
from dataclasses import dataclass
from pyjava.api.mlsql import PythonContext,RayContext
from byzerllm.apps.byzer_storage import env as env_detect
from byzerllm.utils.connect_ray import _check_java_version

@dataclass
class Env:
    llm:ByzerLLM
    retrieval:Optional[ByzerRetrieval]
    ray_context:RayContext           
     

def prepare_env(globals_info:Dict[str,Any],context:PythonContext)->Env:
    
    conf:Dict[str,str] = context.conf
    ray_address = conf.get("rayAddress","auto")                    
    ray_context = RayContext.connect(globals_info,ray_address)
        
    retrieval = None    
        
    llm = byzerllm.ByzerLLM()    
    return Env(llm=llm,
            retrieval=retrieval,
            ray_context=ray_context)

def chat(ray_context):   
    conf = ray_context.conf()
    udf_name = conf["UDF_CLIENT"] 
    
    input_value = [json.loads(row["value"]) for row in ray_context.python_context.fetch_once_as_rows()]
    
    llm = byzerllm.get_single_llm(udf_name,product_mode="lite")
    
    result = []
    for value in input_value:
        v = value.get("query",value.get("instruction",""))
        history = json.loads(value.get("history","[]"))
        
        for key in ["query","instruction","history","user_role","system_msg","assistant_role"]:
            value.pop(key,"")            
                
        conversations = history + [{
            "role":"user",
            "content":v
        }]
        t = llm.chat_oai(conversations=conversations,llm_config={**value})

        response = asdict(t[0])
        
        new_history =  history + [{
            "role":"user",
            "content":v
        }] + [{
            "role":"assistant",
            "content":response["output"]
        }]  

        response["history"] = new_history    
        
        result.append({"value":[json.dumps(response,ensure_ascii=False)]})
    
    ray_context.build_result(result) 


def deploy(infer_params:str,conf:Dict[str,str]):
    '''
    !byzerllm setup single;
    !byzerllm setup "num_gpus=4";
    !byzerllm setup "resources.master=0.001";
    run command as LLM.`` where 
    action="infer"
    and pretrainedModelType="llama"
    and localModelDir="/home/byzerllm/models/openbuddy-llama-13b-v5-fp16"
    and reconnect="true"
    and udfName="llama_13b_chat"
    and modelTable="command";
    '''
    infer_params = json.loads(infer_params)
    llm = ByzerLLM()
    
    num_gpus = float(conf.get("gpus_per_worker",conf.get("gpusPerWorker",0)))
    num_cpus = float(conf.get("cpus_per_worker",conf.get("cpusPerWorker",0.001)))
    num_workers = int(conf.get("num_workers",conf.get("numWorkers",1)))
    worker_concurrency = int(conf.get("worker_concurrency",conf.get("workerConcurrency",1)))

    pretrained_model_type = infer_params.get("pretrainedModelType","custom/auto")
    model_path = infer_params.get("localModelDir","")
    
    infer_params.pop("pretrainedModelType","")
    infer_params.pop("localModelDir","")
    infer_params.pop("udfName","")
    infer_params.pop("modelTable","")
    infer_params.pop("reconnect","")

    chat_name = conf["UDF_CLIENT"]
    
    llm.setup_num_workers(num_workers).setup_worker_concurrency(worker_concurrency)
    llm.setup_cpus_per_worker(num_cpus).setup_gpus_per_worker(num_gpus)

    llm.deploy(model_path=model_path,
            pretrained_model_type=pretrained_model_type,
            udf_name=chat_name,
            infer_params={
               **infer_params
            })            


