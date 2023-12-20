import ray
import json
import os
import uuid
import shutil
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from typing import Dict, Any,List,Generator
from pyjava.storage import streaming_tar as STar
from pyjava import RayContext
from pyjava.api.mlsql import DataServer
from byzerllm import BlockRow
from ray.air.util.torch_dist import (
    ActorHandle,
    _get_node_and_gpu_ids,
    _init_torch_distributed,
    get_address_and_port,
)
from . import qlora as QLoraTrainer
from byzerllm import restore_model
from .. import print_flush

DEFAULT_QLORA_CONFIG = {
    'output_dir': '',
    'model_name_or_path': '',
    'train_file': '',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 16,
    'learning_rate': 0.0002,
    'max_seq_length': 1024,
    'logging_steps': 300,
    'save_steps': 500,
    'save_total_limit': 1,
    'lr_scheduler_type': 'cosine',
    'warmup_steps': 3000,
    'lora_rank': 64,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'gradient_checkpointing': False,
    'disable_tqdm': False,
    'optim': 'paged_adamw_32bit',
    'seed': 42,
    'fp16': True,
    'report_to': 'tensorboard',
    'dataloader_num_workers': 0,
    'save_strategy': 'steps',
    'weight_decay': 0,
    'max_grad_norm': 0.3,
    'remove_unused_columns': False
 }

@ray.remote
class SFT:
    def __init__(self,data_refs:List[DataServer],sft_config:Dict[str,Any],train_params:Dict[str,str],sys_conf: Dict[str, str]) -> None:
        if "runIn" in sys_conf and sys_conf["runIn"] == "driver":
            raise Exception('''
                SFT can not run in driver. 
                Try the one of the following instructions:

                1. !byzerllm setup sft; 
                2. !byzerllm setup "runIn=executor"
            ''')
             
        self.sft_config = sft_config
        self.data_refs = data_refs
        self.train_params = train_params
        self.sys_conf = sys_conf

    def setup_tensorboard(self)->Optional[Tuple[str,int]]:        
        logging_dir = self.sft_config["logging_dir"]
        import subprocess               
        ip, port = get_address_and_port()
        log_dir = logging_dir            
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tb_process = subprocess.Popen(['tensorboard', '--logdir', log_dir,"--port",str(port),"--host",ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)                                    
        self.tensorboard_pid = tb_process.pid
        return (ip,port)

    def train(self,args:List[str]):
        
        sft_name = self.train_params["name"] if "name" in self.train_params else f"sft-{self.sys_conf['OWNER']}"
        print_flush(f"[{sft_name}] SFT Config: {self.sft_config}")

        if not os.path.exists(self.sft_config["output_dir"]):
            os.makedirs(self.sft_config["output_dir"])
        
        train_file = self.sft_config["train_file"]
        if not os.path.exists(train_file): 
            os.makedirs(os.path.dirname(train_file))

        if "localModelDir" not in self.train_params:
            restore_model(self.conf,self.sft_config["model_name_or_path"])                                  
        
        
        # prepare data
        if self.data_refs:
            with open(train_file,"w") as f: 
                count = 0
                for item in RayContext.collect_from(self.data_refs):                
                    if "conversation" in item:
                        item["conversation"] = item["conversation"].tolist()
                        s = json.dumps(item,ensure_ascii=False)               
                        f.write(s+"\n")                    
                    elif "instruction" in item and "output" in item :
                        # support alpaca format data
                        history = item.get("history",[]) 
                        
                        if hasattr(history,"tolist"):
                            history = history.tolist()

                        input = item.get("input","")
                        conversation = [sub.tolist() for sub in history]
                        conversation = [{"human":x[0],"assistant":x[1]} for x in conversation]
                        latest_conversation = [{"human":item["instruction"]+"\n"+input,"assistant":item["output"]}] if "instruction" in item and item["instruction"] else []
                        s = json.dumps({
                            "category":"",
                            "conversation":conversation + latest_conversation,
                            "conversation_id":count,
                            "dataset":"",                
                        },ensure_ascii=False)               
                        f.write(s+"\n") 
                    else:
                        raise Exception(f"Unknown data format: {item}")                                            
                    count += 1       
        
        ip,port = self.setup_tensorboard()
        print_flush(f"[{sft_name}] Tensorboard is running at: {ip}:{port}")

        final_path = QLoraTrainer.train(json.dumps(self.sft_config,ensure_ascii=False), args, {
            "model_type": self.train_params.get("model_type","casual_lm"),"sft_name":sft_name
        })
        # copy the pretrained model to output dir
        if self.train_params.get("skipCopyPretrainedModel","false") == "false":
            print_flush(f'[{sft_name}] Copy pretrained model: {self.sft_config["model_name_or_path"]} to {os.path.join(final_path,"pretrained_model")}')        
            shutil.copytree(self.sft_config["model_name_or_path"],os.path.join(final_path,"pretrained_model"))
        
        # if detached, do not transfer the model to delta lake
        detached = self.train_params.get("detached","false") == "true"
        if detached:
            print_flush(f'''
              [{sft_name}] Train Actor is already finished. You can check the model in: {final_path}              
              ''') 
            return ([],0)
        
        # push the model to ray object store
        result = []
        count = 0
        print_flush(f"[{sft_name}] Store model({final_path}) to Ray object store")
        for item in STar.build_rows_from_file(final_path):
            if count % 1000 == 0:
                print_flush(f"[{sft_name}] Progress: {count} processed")
            count += 1    
            result.append(ray.put(item))
        
        print_flush(f'''
              [{sft_name}] Train Actor already finished.
              [{sft_name}] It may take a while to transfer the model from Ray object store to delta lake. 
              [{sft_name}] Try to check the progress in Byzer console or Byzer Notebook. 
              ''')    
        return (result,count) 

def sft_train(data_refs:List[DataServer],train_params:Dict[str,str],sys_conf: Dict[str, str])->Generator[BlockRow,Any,Any]:
    
    localPathPrefix = train_params.get("localPathPrefix","/tmp/byzerllm")
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d-%H-%M-%S")
    sft_name = train_params["name"] if "name" in train_params else f"sft-{sys_conf['OWNER']}-{formatted_time}"        
    
    rd = f"{sft_name}-{str(uuid.uuid4())}"    
    
    model_dir = os.path.join(localPathPrefix,rd,"pretrained_model")

    if "localModelDir" in train_params:
        model_dir = train_params["localModelDir"]

    output_dir = os.path.join(localPathPrefix,rd,"finetune_model")
    logging_dir = os.path.join(localPathPrefix,rd,"logging")
    data_dir = os.path.join(localPathPrefix,rd,"finetune_data")
    
    if "data_dir" in train_params:
        data_dir = train_params["data_dir"]

    data_file = os.path.join(data_dir,"data.jsonl")

    train_worker_conf = {}
    if "num_cpus" in sys_conf:
        train_worker_conf["num_cpus"] = float(sys_conf["num_cpus"])

    if "num_gpus" in sys_conf:
        train_worker_conf["num_gpus"] = float(sys_conf["num_gpus"])

    custom_resources = [(key.split("resource.")[1], float(sys_conf[key])) for key in
                        sys_conf.keys() if
                        key.startswith("resource.")]

    if len(custom_resources) > 0:
        train_worker_conf["resources"] = dict(custom_resources)   
    
    train_params_sft = {}
    
    for k,v in train_params.items():
        if k.startswith("sft."):
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
            train_params_sft[new_k] = new_v

    if "config" in train_params:
        train_params_sft = {**json.loads(train_params["config"]),**train_params_sft}

    sft_config = {
       **DEFAULT_QLORA_CONFIG,
       **train_params_sft,
       **{
           "output_dir":output_dir,
           "logging_dir": logging_dir,
           "model_name_or_path":model_dir,
           "train_file":data_file,
       }
    }         
    
    detached = train_params.get("detached","true") == "true"
    
    if detached:
        print_flush(f"[{sft_name}] Detached mode is enabled. ")
        train_actor = SFT.options(name=sft_name,lifetime="detached", **train_worker_conf).remote(data_refs,sft_config,train_params,sys_conf)
        train_actor.train.remote([])
        return [] 

    train_actor = SFT.options(name=sft_name,**train_worker_conf).remote(data_refs,sft_config,train_params,sys_conf)
    try:        
        items,obj_count = ray.get(train_actor.train.remote([]))
    except Exception as e:
        ray.kill(train_actor)
        raise e  
            
    print_flush(f"[{sft_name}] Transform Model from Ray object store to new storage(delta lake), total refs: {obj_count}. ")
    count = 0
    for item in items:
        if count % 1000 == 0:
            print_flush(f"[{sft_name}] Process: {float(count)/obj_count*100}%")
        count += 1
        yield ray.get(item)
        