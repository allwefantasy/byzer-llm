from typing import List, Optional, Tuple,Any,Dict,Callable,Generator
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import ray
import torch
import deepspeed
import deepspeed.comm as dist
import sentencepiece as spm
import numpy as np
import datetime
import uuid
import json
import os
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
import dataclasses
from ... import print_flush
import shutil
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME

DEFUALT_CONFIG = '''
{
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1,
  "prescale_gradients": false,
  "zero_allow_untested_optimizer": true,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-8,
      "eps": 1.0e-8,
      "betas": [
        0.9,
        0.95
      ],
      "weight_decay": 0.1
    }
  },
  "tensorboard": {
    "enabled": true    
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
         "device": "cpu"         
     },           
    "offload_param": {
         "device": "cpu"
    },
    "contiguous_gradients": true,
    "allgather_bucket_size": 1e8,
    "reduce_bucket_size": 1e8,
    "overlap_comm": true,
    "reduce_scatter": true
  },
  "steps_per_print": 16,
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": true,
  "bf16": {
    "enabled": true
  }
}
'''

@dataclasses.dataclass
class TrainArgs:
    model_path: str = "" 
    tokenizer_path: str = ""
    sft_name: str = ""
    steps_per_epoch: int = 4096
    is_partition_data: bool = False
    epoches:int = 1
    checkpoint_saving_path: str = "/home/byzerllm/data/checkpoints"
    max_length: int = 4096
    data_dir: str = "/home/byzerllm/data/raw_data"
    data_mode: str = "auto"
     

@dataclasses.dataclass
class DeviceID:
    node_id: int
    gpu_ids: List[int]
    rank: int

class DataEngine():
    def __init__(self, data_dir, tokenizer_path, micro_batch_size, max_length,world_size,rank):
        self.MIN_TEXT_LEN = 20
        self.EOS_TOKEN_ID = 2
        self.data_dir = data_dir
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        self.micro_batch_size = micro_batch_size
        self.max_length = max_length
        self.data = []
        self.global_input_paths = [self.data_dir + "/" + x
                                   for x in os.listdir(self.data_dir)]
        self.local_input_paths = [x for i, x in
                                  enumerate(self.global_input_paths)
                                  if i % world_size == rank]

    def load_data(self):
        for file_path in self.local_input_paths:
            data = []
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line_id, line in enumerate(f):
                    cc = self.sp.EncodeAsIds(line.strip()) + [self.EOS_TOKEN_ID]
                    if len(cc) < self.MIN_TEXT_LEN:
                        cc = []
                    data.extend(cc)
                    if len(data) >= self.micro_batch_size * (self.max_length + 1):
                        index = self.micro_batch_size * (self.max_length + 1)
                        self.data.append(data[:index])
                        data = []
        return
    
    def reset(self):
        self.data = []
        self.load_data()

    def get_data(self):
        data = self.data.pop(0)
        seq = np.asarray(data).reshape(self.micro_batch_size, self.max_length + 1)
        data = torch.LongTensor(seq)
        data = data.cuda(non_blocking=True)
        return data

class ParallelConfig:
    """Configuration for the distributed execution.    
    """

    def __init__(
        self,
        num_workers:int,            
        get_model:Callable[[str,Dict],Any],        
        ds_config:Dict[Any,Any],         
        data_refs:List[DataServer] = [],
        train_args = TrainArgs(),            
        backend: str = "nccl",  
        setup_nccl_socket_ifname_by_ip:bool = False
    ) -> None:
        self.world_size = num_workers        
        self.backend = backend
        self.ds_config = ds_config if ds_config else json.loads(DEFUALT_CONFIG)
        self.train_args = train_args  
        self.get_model = get_model 
        self.data_refs = data_refs 
        # if the nodes in cluster  have different network interface name, we need to set the NCCL_SOCKET_IFNAME 
        # manually otherwise you may meet the following error in deepspeed:
        # torch.distributed.DistBackendError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1275, 
        # internal error, NCCL version 2.14.3
        # ncclInternalError: Internal check failed.
        # Last error:
        # Proxy Call to rank 8 failed (Connect)
        self.setup_nccl_socket_ifname_by_ip = setup_nccl_socket_ifname_by_ip    
    
def _init_distributed_environment(
        parallel_config: ParallelConfig,
        rank: int,
        distributed_init_method: str        
    ) -> None:        
        if parallel_config.backend == "nccl":
            # Same as in Ray Train
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
            # All workers on a same node should share the same set of
            # visible GPUs. Otherwise they can't talk among themselves.
            # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gid) for gid in gpu_ids)
            if "NCCL_SOCKET_IFNAME" not in os.environ:
                os.environ["NCCL_SOCKET_IFNAME"] = DEFAULT_NCCL_SOCKET_IFNAME

        # os.environ["RANK"] = str(rank)
        # os.environ["LOCAL_RANK"] = str(rank)
        # os.environ["WORLD_SIZE"] = str(parallel_config.world_size)
        # os.environ["LOCAL_WORLD_SIZE"] = str(parallel_config.world_size)        
        print(f'''
deepspeed worker config:
              RANK:{rank} 
              WORLD_SIZE:{parallel_config.world_size}
              CUDA_VISIBLE_DEVICES:{os.environ["CUDA_VISIBLE_DEVICES"]} 
              LOCAL_RANK:{os.environ["LOCAL_RANK"]}
              LOCAL_WORLD_SIZE:{os.environ["LOCAL_WORLD_SIZE"]}
              NCCL_SOCKET_IFNAME：{os.environ["NCCL_SOCKET_IFNAME"]}
''',flush=True) 
        # torch.cuda.set_device(rank)
        """Initialize the distributed environment."""
        deepspeed.init_distributed(
            dist_backend="nccl",
            auto_mpi_discovery=False,
            verbose=True,
            init_method=distributed_init_method,
            rank=rank,
            world_size=parallel_config.world_size,
        )
        # torch.distributed.init_process_group(
        #     backend="nccl",
        #     world_size=parallel_config.world_size,
        #     rank=rank,
        #     init_method=distributed_init_method,            
        # )
        # # A small all_reduce for warmup.
        # torch.distributed.all_reduce(torch.zeros(1).cuda())

        

class ResourceWorker:
    def __init__(
        self,        
        parallel_config: ParallelConfig,        
        rank: int               
    ) -> None:
        self.parallel_config = parallel_config        
        self.rank = rank        
        self.ds_config = self.parallel_config.ds_config

    def get_node_and_gpu_ids(self):
        """Returns the node and GPU ids of the current worker."""
        node_id, gpu_ids = _get_node_and_gpu_ids()
        return DeviceID(node_id, gpu_ids, self.rank)  

    def rank(self):
        return self.rank  
    
    def get_node_ip_address(self):
        return ray.util.get_node_ip_address()
    
    def get_address_and_port(self):
        return get_address_and_port()
    
    def get_network_interface(self):
        import netifaces
        interfaces = netifaces.interfaces()
        target_iface = ""
        for iface in interfaces:
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                ip = addrs[netifaces.AF_INET][0]['addr']
                address = self.get_node_ip_address()
                if ip == address:
                    target_iface = iface
                    break
        return target_iface        


class Worker:
    
    def __init__(
        self,        
        parallel_config: ParallelConfig,        
        rank: int,
        distributed_init_method:str                
       
    ) -> None:
        self.parallel_config = parallel_config        
        self.rank = rank        
        self.ds_config = self.parallel_config.ds_config
        self.get_model = self.parallel_config.get_model
        self.distributed_init_method = distributed_init_method 
        self.data_dir = os.path.join(self.parallel_config.train_args.data_dir,f"data-{self.rank}") 
        
        # if the data is not from data_refs(from Byzer) , it may
        # means that the data is prepared in every node before run the training.
        # we just respect the data_dir provied by the user.                
        if not self.parallel_config.data_refs:                  
            self.data_dir = self.parallel_config.train_args.data_dir

        self.model = None
        self.tokenizer = None
        self.tensorboard_pid = None        
    
    def get_node_and_gpu_ids(self):
        """Returns the node and GPU ids of the current worker."""
        node_id, gpu_ids = _get_node_and_gpu_ids()
        return DeviceID(node_id, gpu_ids, self.rank)        
    
    def setup_tensorboard(self)->Optional[Tuple[str,int]]:
        #         "tensorboard": {
        #     "enabled": true,
        #     "output_path": "/home/byzerllm/data/train_ck/logs/",
        #     "job_name": "7b-pt"
        # },
        tensorboard_config = self.ds_config.get("tensorboard", {"enabled":False})
        if tensorboard_config["enabled"]:   
            import subprocess               
            ip, port = get_address_and_port()
            log_dir = tensorboard_config["output_path"]
            job_name = tensorboard_config["job_name"]
            log_dir = os.path.join(log_dir,job_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            tb_process = subprocess.Popen(['tensorboard', '--logdir', log_dir,"--port",str(port),"--host",ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE)                                    
            self.tensorboard_pid = tb_process.pid
            return (ip,port)
        return None

    def _train(self,data_engine, model_engine):
        model_engine.train()
        step = 0
        while step < self.parallel_config.train_args.steps_per_epoch:
            data = data_engine.get_data()
            loss = model_engine(data, labels=data).loss
            model_engine.backward(loss)
            model_engine.step()
            step += 1
        return  

    def get_checkpoint(self):
        if self.rank == 0:
            sft_name = self.parallel_config.train_args.sft_name
            final_path = self.parallel_config.train_args.checkpoint_saving_path
            # get the last checkpoint
            # the checkpoint path is like this:
            # /home/byzerllm/data/sft-20230805-1224-30-173a8dca-9e4a-411c-9fcb-fc979e3460f6/finetune_model/Epoch-1
            # /home/byzerllm/data/sft-20230805-1224-30-173a8dca-9e4a-411c-9fcb-fc979e3460f6/finetune_model/Epoch-2
            # get the last one
            dirs = os.listdir(final_path)
            dirs.sort(key=lambda x: int(x.split("-")[-1]))
            final_path = os.path.join(final_path,dirs[-1])

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
        return None


    
    def train(self):        
        data_engine = self.prepare_data()
        model_engine = self.prepare_model()        
        epoch = 0
        while epoch < self.parallel_config.train_args.epoches:
            self._train(data_engine, model_engine)
            epoch += 1
            model_engine.save_checkpoint(f"{self.parallel_config.train_args.checkpoint_saving_path}",
                                        tag=f"Epoch-{epoch}")
            data_engine.reset()
            
    def prepare_data(self):        
        
        if self.parallel_config.data_refs: 
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

            train_file = os.path.join(self.data_dir,f"train.txt")
            
            '''
            simplely write data to text file
            may need to be think how to handle for new line if the conversation contains new line. This is because
            the data_engine will get data line by line util touch the limit of max_length, then this seuqence will be
            used to train the model.
            But since this is for pretraining, it should be fine.
            '''
            with open(train_file,"w") as f: 
                count = 0
                data_ref = self.parallel_config.data_refs[self.rank]
                print(f"Start to read data to {data_ref.host}:{data_ref.port}. target file:{train_file}",flush=True)
                for item in RayContext.collect_from([data_ref]):                
                    if "conversation" in item:
                        item["conversation"] = item["conversation"].tolist()
                        s =  " ".join(conversation)
                        f.write(s+"\n")                    
                    elif "history" in item:
                        # support alpaca format data
                        conversation = [sub.tolist() for sub in item["history"].tolist()]
                        conversation = [{"human":x[0],"assistant":x[1]} for x in conversation]
                        latest_conversation = [{"human":item["instruction"],"assistant":item["output"]}] if "instruction" in item and item["instruction"] else []
                        s = " ".join(conversation) + " ".join(latest_conversation)
                        f.write(s+"\n")
                    elif "text" in item:
                        f.write(item["text"]+"\n")
                    else:
                        raise Exception("Unknow data format")                             
                    count += 1         

        
        tokenizer_path = self.parallel_config.train_args.tokenizer_path        
        micro_batch_size = self.ds_config["train_micro_batch_size_per_gpu"]
        max_length = self.parallel_config.train_args.max_length

        world_size = 1 if self.parallel_config.train_args.is_partition_data else self.parallel_config.world_size
        rank = 0 if self.parallel_config.train_args.is_partition_data else self.rank

        data_engine = DataEngine(self.data_dir, tokenizer_path, micro_batch_size, max_length,world_size,rank)
        data_engine.load_data()
        return data_engine     
    
    def prepare_model(self):
        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method)
        
        # check the enabled parameter here: https://github.com/microsoft/DeepSpeed/issues/3234
        
        with deepspeed.zero.Init(config_dict_or_path=self.ds_config,
                             enabled=self.ds_config["zero_optimization"]["stage"] == 3,
                             mem_efficient_linear=False,
                             mpu=None):
            model = self.get_model()            
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            model_engine, _, _, _ = deepspeed.initialize(model=model,
                                                         config=self.ds_config,
                                                        optimizer=None,
                                                        model_parameters=model_parameters)
            return model_engine
            
           

class DeepSpeedTrain:
    def __init__(self,parallel_config: ParallelConfig):    

        

        # get resource manually. If use num_gpus, the ray will set cuda_visible_devices automatically, and this
        # will cause the deepspeed can't get the right gpu_ids enven if we set the cuda_visible_devices manually.                
        resource_workers = [] 
        workers = []       
        
        for rank in range(parallel_config.world_size):    
            worker_cls = ResourceWorker                        
            worker_cls = ray.remote(
                        num_cpus=0,
                        num_gpus=1,
                        name=f"RW-{rank}-{parallel_config.train_args.sft_name}",
                        # resources={f"node:{master_addr}": 1e-3},
                        # runtime_env=runtime_env,                        
                    )(worker_cls).remote
            worker = worker_cls(parallel_config,rank)
            resource_workers.append(worker)

        master_addr, master_port = ray.get(resource_workers[0].get_address_and_port.remote())          
        distributed_init_method = f"tcp://{master_addr}:{master_port}"  
        print(f"deepspeed: master_addr:{master_addr},master_port:{master_port}",flush=True)

        self.resource_workers  = resource_workers        
        self.node_id_to_workers = {}
        self.node_id_to_gpus = {}
        self.node_id_to_nccl_socket_device = {}

        for resource_worker in self.resource_workers:
            device = ray.get(resource_worker.get_node_and_gpu_ids.remote())            
     
            if device.node_id not in self.node_id_to_workers:
                self.node_id_to_workers[device.node_id] = []
            
            if device.node_id not in self.node_id_to_gpus:
                self.node_id_to_gpus[device.node_id] = []    
            
            self.node_id_to_workers[device.node_id].append(resource_worker)    
            self.node_id_to_gpus[device.node_id].extend(device.gpu_ids)
            self.node_id_to_gpus[device.node_id].sort()

            if parallel_config.setup_nccl_socket_ifname_by_ip:
                self.node_id_to_nccl_socket_device[device.node_id] = ray.get(resource_worker.get_network_interface.remote())

        for node_id, resource_workers in self.node_id_to_workers.items():
            for local_rank,resource_worker in enumerate(resource_workers):
                rank = ray.get(resource_worker.rank.remote()) 
                worker_cls = Worker  
                gpu_ids = self.node_id_to_gpus[node_id]
                nccl_socket_ifname = DEFAULT_NCCL_SOCKET_IFNAME
                if parallel_config.setup_nccl_socket_ifname_by_ip:
                    nccl_socket_ifname = self.node_id_to_nccl_socket_device[node_id]
                env_vars = {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES":"true",
                        "CUDA_VISIBLE_DEVICES": ",".join([str(gid) for gid in gpu_ids]),
                        "LOCAL_RANK": str(local_rank),
                        "RANK": str(rank),
                        "NCCL_SOCKET_IFNAME": f"={nccl_socket_ifname}",
                        "LOCAL_WORLD_SIZE": str(len(gpu_ids)),
                        "WORLD_SIZE": str(parallel_config.world_size)
                        }                                        
                runtime_env = {"env_vars": env_vars}  
                node_ip = ray.get(resource_worker.get_node_ip_address.remote())  
                worker_cls = ray.remote(
                            num_cpus=0, 
                            num_gpus=0,
                            name=f"W-{rank}-{parallel_config.train_args.sft_name}",
                            resources={f"node:{node_ip}": 1e-3},
                            runtime_env=runtime_env,                        
                        )(worker_cls).remote
                worker = worker_cls(parallel_config,rank,distributed_init_method)
                workers.append(worker)  
                if rank == 0:
                    addr_port = ray.get(worker.setup_tensorboard.remote())
                    if addr_port:
                        print(f"tensorboard: http://{addr_port[0]}:{addr_port[1]}",flush=True)

        self.workers = workers                              
    
              
    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method)
            executor = executor.remote
            output = executor(*args, **kwargs)
            all_outputs.append(output)

        all_outputs = ray.get(all_outputs)            
        
        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output 

class DeepSpeedTrainer:
    def __init__(self,name:str) -> None:
        self.sft_name = name
        self.dst = None 
        self.output_dir = None

    def get_checkpoint_path(self):
        return self.output_dir    

    def sfft_train(self,data_refs:List[DataServer],train_params:Dict[str,str],sys_conf: Dict[str, str]):                

        localPathPrefix = train_params.get("localPathPrefix","/tmp/byzerllm")

        sft_name = self.sft_name
        rd = f"{sft_name}-{str(uuid.uuid4())}"        

        num_gpus = int(sys_conf.get("num_gpus",0))
        
        assert num_gpus > 0, 'num_gpus must be greater than 0. Try to fix it with `!byzerllm setup "num_gpus=4"`'
        
        is_partition_data = len(data_refs) != 0

        if is_partition_data:
            assert num_gpus == len(data_refs), f'''The number of data refs({len(data_refs)}) must be equal to the number of GPUs({num_gpus}).
            Try to fix it with `!byzerllm setup "num_gpus={len(data_refs)}"` or repartition the data with the following command:
            
            ```
            run oldTable as TableRepartition.`` where partitionNum="{num_gpus}" as newTable;
            ```
            Notice that make sure Byzer engine have CPUs more than {num_gpus}.
            '''

        data_dir = train_params["localDataDir"] if "localDataDir" in train_params else os.path.join(localPathPrefix,rd,"finetune_data")
        output_dir = os.path.join(localPathPrefix,rd,"finetune_model")
        self.output_dir = output_dir
        tensorboard_dir = os.path.join(localPathPrefix,rd,"tensorboard_dir")
        model_dir = os.path.join(localPathPrefix,rd,"pretrained_model")
        
        if "localModelDir" in train_params:
            model_dir = train_params["localModelDir"]

        pretrained_model_type = train_params.get("pretrainedModelType","")
        if "/" in  pretrained_model_type:
            pretrained_model_type = pretrained_model_type.split("/")[-1]
        
        def get_model():
            if pretrained_model_type == "llama2":            
                return AutoModelForCausalLM.from_pretrained(model_dir,
                                                            trust_remote_code=True,
                                                            ignore_mismatched_sizes=True)
            else:
                return AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True)
        
        setup_nccl_socket_ifname_by_ip = False
        if "sfft.bool.setup_nccl_socket_ifname_by_ip" in train_params:
            setup_nccl_socket_ifname_by_ip = train_params["sfft.bool.setup_nccl_socket_ifname_by_ip"] == "true"
        
        tokenizer_path = train_params["sfft.str.tokenizer_path"] if "sfft.str.tokenizer_path" in train_params else f"{model_dir}/tokenizer.model"        
        max_length = int(train_params.get("sfft.int.max_length",4096))
        epoches = int(train_params.get("sfft.int.epoches",1))
        steps_per_epoch = int(train_params.get("sfft.int.steps_per_epoch",10))
        
        try:
            ds_config=  json.loads(train_params.get("deepspeedConfig",DEFUALT_CONFIG))
        except Exception as e:        
            print(f'deepspeedConfig is not a valid json string:\n{train_params.get("deepspeedConfig","{}")}',flush=True)
            print(f"Byzer-LLM will ues the default deepspeed config:\n{DEFUALT_CONFIG}",flush=True)
            ds_config = json.loads(DEFUALT_CONFIG)
            

        if "tensorboard"  in ds_config and  ds_config["tensorboard"].get("enabled",False):
            if "output_path" not in ds_config["tensorboard"]:
                ds_config["tensorboard"]["output_path"] = tensorboard_dir
                ds_config["tensorboard"]["job_name"] = sft_name

        print(f'''
    Train Configuration:
        pretrained_model_type:{pretrained_model_type} 
        model_dir:{model_dir} 
        output_dir:{output_dir}
        data_dir:{data_dir}
        is_partition_data:{is_partition_data}
        max_length:{max_length}
        epoches:{epoches}
        steps_per_epoch:{steps_per_epoch} 
        setup_nccl_socket_ifname_by_ip:{setup_nccl_socket_ifname_by_ip}   
        num_gpus:{num_gpus}            
            ''',flush=True)   
        

        dst = DeepSpeedTrain(ParallelConfig(
        data_refs = data_refs,
        num_workers = num_gpus,
        get_model = get_model,
        ds_config = ds_config,     
        setup_nccl_socket_ifname_by_ip = setup_nccl_socket_ifname_by_ip,
        train_args=TrainArgs(
            model_path=model_dir,
            tokenizer_path = tokenizer_path,
            data_dir = data_dir,  
            checkpoint_saving_path = output_dir,   
            steps_per_epoch = steps_per_epoch,
            max_length = max_length,
            epoches=epoches,
            is_partition_data = is_partition_data,
            sft_name=sft_name
            )
        ))

        self.dst = dst
        ray.get([worker.train.remote() for worker in dst.workers])
        if train_params.get("detached","true") == "true":
            return [],0
        chunks,count = ray.get(dst.workers[0].get_checkpoint.remote())
        return chunks,count


def sfft_train(data_refs:List[DataServer],train_params:Dict[str,str],sys_conf: Dict[str, str])->Generator[BlockRow,Any,Any]:
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d-%H-%M-%S")
    sft_name = train_params["name"] if "name" in train_params else f"sft-{sys_conf['OWNER']}-{formatted_time}"        

    detached = train_params.get("detached","true") == "true"
    options = {"name":sft_name}
    if detached:        
        options["lifetime"] = "detached"
                
    worker_cls = ray.remote(**options)(DeepSpeedTrainer).remote
    trainer = worker_cls(name=sft_name)

    if detached:
        print_flush(f"[{sft_name}] Detached mode is enabled. ")
        trainer.sfft_train.remote(data_refs,train_params,sys_conf)        
        return []
        
    chunks,obj_count = ray.get(trainer.sfft_train.remote(data_refs,train_params,sys_conf))    
    checkpoint_path = ray.get(trainer.get_checkpoint_path.remote())
    node = ray.get(trainer.dst.resource_workers[0].get_node_ip_address.remote())
    print_flush(f"The model is finised training, Please check the path: {node}:{checkpoint_path}")
    
    if obj_count == 0:    
        return []

    print_flush(f"[{sft_name}] Transform Model from Ray object store to new storage(delta lake), total refs: {obj_count}. ")
    count = 0
    for item in chunks:
        if count % 1000 == 0:
            print_flush(f"[{sft_name}] Process: {float(count)/obj_count*100}%")
        count += 1
        yield ray.get(item)

