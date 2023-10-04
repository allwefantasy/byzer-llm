from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import ray
import torch
import os
import ray
from typing import Any,Any,Dict, List,Tuple,Generator,Optional,Union
import types

from pyjava.api.mlsql import DataServer
from byzerllm.utils.metrics import Metric
from .. import BlockRow


INFERENCE_NAME = "auto"
INFER_TOKEN_METRICS = Metric()

def stream_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=1024, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = tokenizer(ins, return_token_type_ids=False,return_tensors="pt").to(device)
    response = self.generate(
        input_ids=tokens["input_ids"],
        max_new_tokens=max_length,
        repetition_penalty=1.05,
        temperature=temperature,
        attention_mask=tokens.attention_mask        
    )
    answer = tokenizer.decode(response[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)
    return [(answer,"")]


def ray_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    from aviary.backend.server.models import Prompt
    model = self
    response = ray.get(model.generate_text.remote(prompt=Prompt(
        prompt=ins,
        use_prompt_format=False
    ),request=None))
    return [(response.generated_text,"")]

async def async_vllm_chat(model,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    from vllm import  SamplingParams
    from vllm.utils import random_uuid
    request_id = random_uuid()
    
    n: int = 1
    best_of: Optional[int] =  kwargs["best_of"] if "best_of" in kwargs else None
    presence_penalty: float = float(kwargs.get("presence_penalty",0.0))
    frequency_penalty: float = float(kwargs.get("frequency_penalty",0.0))
    top_k: int = int(kwargs.get("top_k",-1))
    use_beam_search: bool = kwargs.get("use_beam_search","false") == "true"
    stop: Union[None, str, List[str]] = kwargs["stop"] if "stop" in kwargs else None
    ignore_eos: bool = kwargs.get("ignore_eos","false") == "true"
    max_tokens: int = max_length
    logprobs: Optional[int] = kwargs["logprobs"] if "logprobs" in kwargs else None
    
    sampling_params = SamplingParams(temperature=temperature, 
                                     n = n,
                                     best_of=best_of,
                                     presence_penalty=presence_penalty,
                                     frequency_penalty=frequency_penalty,
                                     top_k=top_k,
                                     use_beam_search=use_beam_search,
                                     stop = stop,
                                     ignore_eos=ignore_eos,
                                     logprobs=logprobs,
                                     top_p=top_p, 
                                     max_tokens=max_tokens)
    
    results_generator = model.generate(ins, sampling_params,request_id) 
    final_output = None
    async for request_output in results_generator:        
        final_output = request_output
    assert final_output is not None    
    
    text_outputs = [output for output in final_output.outputs]
    generated_text = text_outputs[0].text
    
    
    input_tokens_count = len(final_output.prompt_token_ids)
    generated_tokens_count = len(text_outputs[0].token_ids) 
    
    print(f"total_tokens_count:{input_tokens_count + generated_tokens_count} request_id:{final_output.request_id}  input_tokens_count:{input_tokens_count} generated_tokens_count:{generated_tokens_count}",flush=True)    
    INFER_TOKEN_METRICS.inc(f"infer_{INFERENCE_NAME}_input_tokens_num",input_tokens_count,tags={"request_id":final_output.request_id})
    INFER_TOKEN_METRICS.inc(f"infer_{INFERENCE_NAME}_output_tokens_num", generated_tokens_count,tags={"request_id":final_output.request_id})
    INFER_TOKEN_METRICS.push()
    return [(generated_text,"")]   

def block_vllm_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    from vllm import  SamplingParams        
    n: int = 1
    best_of: Optional[int] =  kwargs["best_of"] if "best_of" in kwargs else None
    presence_penalty: float = float(kwargs.get("presence_penalty",0.0))
    frequency_penalty: float = float(kwargs.get("frequency_penalty",0.0))
    top_k: int = int(kwargs.get("top_k",-1))
    use_beam_search: bool = kwargs.get("use_beam_search","false") == "true"
    stop: Union[None, str, List[str]] = kwargs["stop"] if "stop" in kwargs else None
    ignore_eos: bool = kwargs.get("ignore_eos","false") == "true"
    max_tokens: int = max_length
    logprobs: Optional[int] = kwargs["logprobs"] if "logprobs" in kwargs else None

    model = self
    sampling_params = SamplingParams(temperature=temperature, 
                                     n = n,
                                     best_of=best_of,
                                     presence_penalty=presence_penalty,
                                     frequency_penalty=frequency_penalty,
                                     top_k=top_k,
                                     use_beam_search=use_beam_search,
                                     stop = stop,
                                     ignore_eos=ignore_eos,
                                     logprobs=logprobs,
                                     top_p=top_p, 
                                     max_tokens=max_tokens)
    
    outputs = model.generate([ins], sampling_params)    

    output = outputs[0].outputs[0]
    generated_text = output.text
        
    input_tokens_count = len(outputs[0].prompt_token_ids)
    generated_tokens_count = len(output.token_ids) 
    print(f"total_tokens_count:{input_tokens_count + generated_tokens_count} request_id:{outputs[0].request_id} output_num:{len(outputs)}/{len(outputs[0].outputs)}  input_tokens_count:{input_tokens_count} generated_tokens_count:{generated_tokens_count}",flush=True)

    return [(generated_text,"")]   


def vllm_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    model = self        
    return block_vllm_chat(model,model,tokenizer,ins,his,max_length,top_p,temperature,**kwargs)   

def init_model(model_dir,infer_params:Dict[str,str]={},sys_conf:Dict[str,str]={}): 
    infer_mode = sys_conf.get("infer_backend","transformers")
    quatization = infer_params.get("quatization","false") == "true"
    
    if infer_mode == "transformers":
        raise Exception('''
transformers backend is not supported in pretrainedModelType: auto.
Try to use ray/tgi or ray/deepspeed or ray/vllm.

For example:

!byzerllm setup "infer_backend=ray/vllm";
''')
    

    if infer_mode == "tgi":
        import byzerllm.utils.inference as TGI
        return TGI.init_model(model_dir,infer_params)                
    
    if infer_mode in ["aviary/deepspeed","aviary/devicemap"]:   
        num_workers = int(sys_conf.get("num_gpus",1))   
        udfName = infer_params["udfName"]
        mode = infer_mode.split("/")[1]
        from byzerllm.utils.rayinfer import build_model_serving
        model = build_model_serving(udfName,model_dir, mode=mode, num_workers=num_workers)        
        model.stream_chat = types.MethodType(ray_chat, model) 
        return (model,None) 

    if infer_mode == "ray/vllm":        
        num_gpus = int(sys_conf.get("num_gpus",1))
        print(f"infer_mode:{infer_mode} tensor_parallel_size: {num_gpus}")
        global INFERENCE_NAME
        INFERENCE_NAME = infer_params.get("udfName","auto")
        
        # use_np_weights: bool = infer_params.get("backend.use_np_weights","false") == "true"
        # use_dummy_weights: bool = infer_params.get("backend.use_dummy_weights","false") == "true"
        dtype: str = infer_params.get("backend.dtype","auto")
        seed: int = int(infer_params.get("backend.seed",0))
        worker_use_ray: bool = infer_params.get("backend.worker_use_ray","false") == "false"
        pipeline_parallel_size: int = int(infer_params.get("backend.pipeline_parallel_size",1))
        tensor_parallel_size: int = num_gpus
        block_size: int = int(infer_params.get("backend.block_size",16))
        swap_space: int = int(infer_params.get("backend.swap_space",4))  # GiB
        gpu_memory_utilization: float = float(infer_params.get("backend.gpu_memory_utilization",0.90))
        max_num_batched_tokens: int = int(infer_params.get("backend.max_num_batched_tokens",2560))
        max_num_seqs: int = int(infer_params.get("backend.max_num_seqs",256))
        disable_log_stats: bool = infer_params.get("backend.disable_log_stats","false") == "true"

        from vllm.engine.async_llm_engine import AsyncLLMEngine,AsyncEngineArgs     
        engine_args = AsyncEngineArgs(
            engine_use_ray=False,
            disable_log_requests=False,
            model=model_dir,
            tokenizer=None,tokenizer_mode="auto",
            trust_remote_code=True,    
            worker_use_ray=worker_use_ray,                                        
            dtype=dtype,
            seed=seed,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            block_size=block_size,
            swap_space=swap_space,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            disable_log_stats=disable_log_stats
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)                       
        # llm.stream_chat = types.MethodType(vllm_chat, llm) 
        llm.async_stream_chat = types.MethodType(async_vllm_chat, llm) 
        return (llm,None)  

    if  infer_mode == "ray/deepspeed":
        from .backend_ds import DeepSpeedInference,ParallelConfig        
        num_gpus = int(sys_conf.get("num_gpus",1))
        model = DeepSpeedInference(ParallelConfig(num_workers=num_gpus,model_dir=model_dir))    
        return (model,None)                     

    pretrained_model_dir = os.path.join(model_dir,"pretrained_model")
    adaptor_model_dir = model_dir
    is_adaptor_model = os.path.exists(pretrained_model_dir)
    
    if not is_adaptor_model:        
        pretrained_model_dir = model_dir

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)    

    if quatization:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_dir,
            trust_remote_code=True,            
            device_map="auto",
            quantization_config=nf4_config,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir,trust_remote_code=True,
                                                device_map='auto',                                                
                                                torch_dtype=torch.bfloat16                                                
                                                )
    if is_adaptor_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adaptor_model_dir)

    model.eval()  
    if quatization:
        model = torch.compile(model)   

    model.stream_chat = types.MethodType(stream_chat, model)     
    return (model,tokenizer)


def sft_train(data_refs:List[DataServer],
              train_params:Dict[str,str],
              conf: Dict[str, str])->Generator[BlockRow,Any,Any]:
    from ..utils.sft import sft_train as common_sft_train
    return common_sft_train(data_refs,train_params,conf) 

