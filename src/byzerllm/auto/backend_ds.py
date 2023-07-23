from typing import List, Optional, Tuple,Any
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import ray
import torch
import deepspeed
from ray.air.util.torch_dist import (
    ActorHandle,
    _get_node_and_gpu_ids,
    _init_torch_distributed,
    get_address_and_port,
)

class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.        
    """

    def __init__(
        self,
        num_workers:int,
        model_dir:str              
    ) -> None:
        self.world_size = num_workers
        self.model_dir = model_dir
    

DeviceID = Tuple[int, Optional[str], int]

def _init_distributed_environment(
        parallel_config: ParallelConfig,
        rank: int,
        distributed_init_method: str,
    ) -> None:
        
        """Initialize the distributed environment."""
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )
        print(f"deepspeed inference worker:rank:{rank} init_process_group success. ",flush=True)
        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cuda())
        print(f"deepspeed inference worker:rank:{rank} A small all_reduce for warmup",flush=True)

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

        # Initialize the distributed environment.
        _init_distributed_environment(parallel_config, rank,
                                      distributed_init_method)
        
        self.model = None
        self.tokenizer = None
       

    def init_model(self):
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

    def execute_model(self,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokens = self.okenizer(ins, return_token_type_ids=False,return_tensors="pt").to(device)
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
        for rank in range(parallel_config.world_size):    
            worker_cls = Worker
            worker_cls = ray.remote(
                        num_cpus=0,
                        num_gpus=1,
                        resources={f"node:{master_addr}": 1e-3},
                    )(worker_cls).remote
            worker = worker_cls(parallel_config,rank,distributed_init_method)
            workers.append(worker)
        self.workers  = workers  
        [worker.init_model.remote() for worker in self.workers]      

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


  
    
