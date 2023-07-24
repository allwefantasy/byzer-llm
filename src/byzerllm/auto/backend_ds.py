from typing import List, Optional, Tuple,Any
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import ray
import torch
import deepspeed
import os
from ray.air.util.torch_dist import (
    ActorHandle,
    _get_node_and_gpu_ids,
    _init_torch_distributed,
    get_address_and_port,
)
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME

class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.        
    """

    def __init__(
        self,
        num_workers:int,
        model_dir:str,        
        backend: str = "nccl",              
    ) -> None:
        self.world_size = num_workers
        self.model_dir = model_dir
        self.backend = backend
    

DeviceID = Tuple[int, Optional[str], int]

def _init_distributed_environment(
        parallel_config: ParallelConfig,
        rank: int,
        distributed_init_method: str,
        gpu_ids: List[int],
    ) -> None:
        print(f'deepspeed inference worker before::rank:{rank} CUDA_VISIBLE_DEVICES:{os.environ["CUDA_VISIBLE_DEVICES"]}',flush=True)
        if parallel_config.backend == "nccl":
            # Same as in Ray Train
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
            # All workers on a same node should share the same set of
            # visible GPUs. Otherwise they can't talk among themselves.
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gid) for gid in gpu_ids)
            if "NCCL_SOCKET_IFNAME" not in os.environ:
                os.environ["NCCL_SOCKET_IFNAME"] = DEFAULT_NCCL_SOCKET_IFNAME

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(parallel_config.world_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(parallel_config.world_size)        
        print(f'deepspeed inference worker after:rank:{rank} CUDA_VISIBLE_DEVICES:{os.environ["CUDA_VISIBLE_DEVICES"]}',flush=True)
        torch.cuda.set_device(rank)
        """Initialize the distributed environment."""
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,            
        )
        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cuda())

        

class Worker:
    
    def __init__(
        self,        
        parallel_config: ParallelConfig,        
        rank: int,
        distributed_init_method: str,
    ) -> None:
        self.parallel_config = parallel_config        
        self.rank = rank
        self.distributed_init_method = distributed_init_method                
        self.model = None
        self.tokenizer = None
       

    def init_model(self,gpu_ids:List[int]):
        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method,gpu_ids)
        
        print(f"deepspeed inference worker:rank:{self.rank} load model {self.parallel_config.model_dir}",flush=True)
        tokenizer = AutoTokenizer.from_pretrained(self.parallel_config.model_dir,trust_remote_code=True)  
        model = AutoModelForCausalLM.from_pretrained(self.parallel_config.model_dir,trust_remote_code=True)       
        model = model.eval()
    
        ds_engine = deepspeed.init_inference(model,
                                mp_size=self.parallel_config.world_size,
                                dtype=torch.half,
                                replace_method="auto",
                                replace_with_kernel_inject=True)
        self.model = ds_engine.module
        self.tokenizer = tokenizer  
        print(f"deepspeed inference worker:rank:{self.rank} init successfully",flush=True)    

    def execute_model(self,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokens = self.tokenizer(ins, return_token_type_ids=False,return_tensors="pt").to(device)
        response = self.model.generate(
            input_ids=tokens["input_ids"],
            max_new_tokens=max_length,
            repetition_penalty=1.05,
            temperature=temperature,
            attention_mask=tokens.attention_mask        
        )
        answer = self.tokenizer.decode(response[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        return [(answer,"")]  
           

class DeepSpeedInference:
    def __init__(self,parallel_config: ParallelConfig ):    

        master_addr, master_port = get_address_and_port()        
        distributed_init_method = f"tcp://{master_addr}:{master_port}"  
        print(f"deepspeed inference: master_addr:{master_addr},master_port:{master_port}",flush=True)
        workers = []
        gpu_ids = ray.get_gpu_ids()
        gpu_ids_str = ",".join([str(gpu) for gpu in gpu_ids])
        
        for rank in range(parallel_config.world_size):    
            worker_cls = Worker  
            # deepspeed will use rank as the device id, and the 
            # ray will automatically set CUDA_VISIBLE_DEVICES for each worker according to the num_gpus
            # for example, suppose we have 0,1,2,4 gpus, and we have 4 workers, then the CUDA_VISIBLE_DEVICES 
            # for the last worker will be 3, and the deepspeed will use 3 as the device id, which is wrong because
            # he can only see one gpu. So we need to set CUDA_VISIBLE_DEVICES to 0,1,2,3 for each worker.
            runtime_env = {"env_vars": {
              "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES":"true",
              "CUDA_VISIBLE_DEVICES":gpu_ids_str
            }}    
            worker_cls = ray.remote(
                        num_cpus=0,
                        num_gpus=0,
                        resources={f"node:{master_addr}": 1e-3},
                        runtime_env=runtime_env,                        
                    )(worker_cls).remote
            worker = worker_cls(parallel_config,rank,distributed_init_method)
            workers.append(worker)
        self.workers  = workers           
        ray.get([worker.init_model.remote(gpu_ids) for worker in self.workers])

    def stream_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=1024, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):        
        output = self._run_workers("execute_model",ins,his,max_length,top_p,temperature)
        return output
              
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


  
    
