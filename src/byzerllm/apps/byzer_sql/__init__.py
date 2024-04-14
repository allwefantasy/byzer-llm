from typing import Dict
from dataclasses import asdict
from byzerllm.utils.client import ByzerLLM
import json

def chat(ray_context):   
    conf = ray_context.conf()
    udf_name = conf["UDF_CLIENT"] 
    
    input_value = [json.loads(row["value"]) for row in ray_context.python_context.fetch_once_as_rows()]
    
    llm = ByzerLLM()
    llm.setup_template(model=udf_name,template="auto")
    llm.setup_default_emb_model_name("emb")
    llm.setup_default_model_name(udf_name)
    llm.setup_extra_generation_params(udf_name,extra_generation_params={
        "temperature":0.01,
        "top_p":0.99
    })
    
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
    num_gpus = int(conf.get("num_gpus",1))
    num_workers = int(conf.get("maxConcurrency",1))

    pretrained_model_type = infer_params.get("pretrainedModelType","custom/auto")
    model_path = infer_params.get("localModelDir","")
    
    infer_params.pop("pretrainedModelType","")
    infer_params.pop("localModelDir","")
    infer_params.pop("udfName","")
    infer_params.pop("modelTable","")
    infer_params.pop("reconnect","")


    chat_name = conf["UDF_CLIENT"]
    
    llm.setup_num_workers(num_workers).setup_gpus_per_worker(num_gpus)

    llm.deploy(model_path=model_path,
            pretrained_model_type=pretrained_model_type,
            udf_name=chat_name,
            infer_params={
               **infer_params
            })            


