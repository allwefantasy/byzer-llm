from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig,StoppingCriteriaList
import ray
import torch
import os
import ray
import time
from typing import Any,Any,Dict, List,Tuple,Generator,Optional,Union
import types

from pyjava.api.mlsql import DataServer
from byzerllm.utils.metrics import Metric
from .. import BlockRow
from byzerllm.utils import VLLMStreamServer,StreamOutputs,SingleOutput
import asyncio
from byzerllm.utils import compute_max_new_tokens,tokenize_stopping_sequences,StopSequencesCriteria



INFERENCE_NAME = "auto"
INFER_TOKEN_METRICS = Metric()


def stream_chat(self,tokenizer,ins:str, his:List[Dict[str,str]]=[],  
        max_length:int=4090, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    timeout_s = float(kwargs.get("timeout_s",60*5)) 
    skip_check_min_length = int(kwargs.get("stopping_sequences_skip_check_min_length",0))       
    
    tokens = tokenizer(ins, return_token_type_ids=False,return_tensors="pt").to(device)

    stopping_criteria = None
    
    import math

    if "stopping_sequences" in kwargs:        
        stopping_sequences = [torch.tensor(word).to(device) for word in tokenize_stopping_sequences(tokenizer,kwargs["stopping_sequences"].split(","))]    
        input_length = tokens["input_ids"].shape[1]
        stopping_criteria=StoppingCriteriaList([StopSequencesCriteria(
            tokenizer=tokenizer,
            stops=stopping_sequences,
            input_start=input_length,
            skip_check_min_length=skip_check_min_length
            )])

    config = self.config

    max_new_tokens = compute_max_new_tokens(tokens, min(max_length, getattr(config, "model_max_length", max_length))) 

    other_params = {}  
    if "early_stopping" in kwargs:
        other_params["early_stopping"] = bool(kwargs["early_stopping"])

    if "repetition_penalty" in kwargs:
        other_params["repetition_penalty"] = float(kwargs["repetition_penalty"])
    
    start_time = time.monotonic()        
    response = self.generate(
        input_ids=tokens["input_ids"],
        max_new_tokens= max_new_tokens,        
        temperature=temperature,
        top_p=top_p,        
        max_time=timeout_s,
        stopping_criteria=stopping_criteria,
        **other_params
    )    
    time_taken = time.monotonic() - start_time    
    new_tokens = response[0][tokens["input_ids"].shape[1]:]
    print(f"generate took {time_taken} s to complete. tokens/s:{len(new_tokens)/time_taken}",flush=True)
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return [(answer,"")]

async def async_get_meta(model):
     from vllm.engine.async_llm_engine import AsyncLLMEngine,AsyncEngineArgs     
     model:AsyncLLMEngine = model
     config = await model.get_model_config()
     return [{"model_deploy_type":"proprietary",
              "backend":"ray/vllm",
              "support_stream": True,
              "max_model_len":config.max_model_len,
              "architectures":getattr(config.hf_config, "architectures", [])
              }]

async def async_vllm_chat(model,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
    from vllm import  SamplingParams
    from vllm.utils import random_uuid

    stream = kwargs.get("stream",False)
    request_id = random_uuid()
    
    # in future, we should add the cancel logic here
    # first_request = False

    # if "request_id" in kwargs:
    #     request_id = kwargs["request_id"]        
    # else:
    #     request_id = random_uuid()
    #     first_request = True

    # if stream and not first_request:
    #     server = ray.get_actor("VLLM_STREAM_SERVER")
    #     final_output = ray.get(server.get_item.remote(request_id))
        
    #     if isinstance(final_output,str):
    #         return [("",{"metadata":{"request_id":request_id,"status":"running"}})]
        
    #     if final_output is None:
    #         return [("",{"metadata":{"request_id":request_id,"status":"finish"}})]
        
    #     text_outputs = [output for output in final_output.outputs]
    #     generated_text = text_outputs[0].text        
    #     return [(generated_text,{"metadata":{"request_id":final_output.request_id}})]    
    
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
    # repetition_penalty: float = float(kwargs.get("repetition_penalty",1.1))

    other_params = {}
    if "early_stopping" in kwargs:
        other_params["early_stopping"] = bool(kwargs["early_stopping"])
    
    if "repetition_penalty" in kwargs:
        other_params["repetition_penalty"] = float(kwargs["repetition_penalty"]) 

    if "stop_token_ids" in kwargs:
        stop_token_ids = kwargs["stop_token_ids"]
        if isinstance(stop_token_ids,str):
            stop_token_ids = [int(i) for i in stop_token_ids.split(",")]
        else:
            stop_token_ids = kwargs["stop_token_ids"]
        other_params["stop_token_ids"] = stop_token_ids        
        
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
                                     max_tokens=max_tokens,
                                     ** other_params
                                     )    
    
    current_time_milliseconds = int(time.time() * 1000)
        
    if stream:
        server = ray.get_actor("VLLM_STREAM_SERVER")
        async def writer():
            results_generator = model.generate(ins, sampling_params,request_id) 
            async for request_output in results_generator:     
                v = StreamOutputs(outputs=[SingleOutput(text=item.text) for item in request_output.outputs])         
                await server.add_item.remote(request_output.request_id, v)
            # mark the request is done
            await server.mark_done.remote(request_output.request_id)
        asyncio.create_task(writer())
        await server.add_item.remote(request_id, "RUNNING")        
        return [("",{"metadata":{"request_id":request_id,"stream_server":"VLLM_STREAM_SERVER"}})]
        
    results_generator = model.generate(ins, sampling_params,request_id) 
    final_output = None
    first_token_time = current_time_milliseconds
    async for request_output in results_generator:  
        if first_token_time == current_time_milliseconds and request_output.outputs and len(request_output.outputs[0].token_ids)>0:
            first_token_time = int(time.time() * 1000)      
        final_output = request_output
    assert final_output is not None    
    
    text_outputs = [output for output in final_output.outputs]
    generated_text = text_outputs[0].text
    prob = text_outputs[0].cumulative_logprob

    current_time_milliseconds2 = int(time.time() * 1000)
        
    input_tokens_count = len(final_output.prompt_token_ids)
    generated_tokens_count = len(text_outputs[0].token_ids) 
    time_cost = current_time_milliseconds2-current_time_milliseconds
    print(f"cost: {time_cost}ms first_token:{first_token_time-current_time_milliseconds}ms speed: {float(generated_tokens_count)/time_cost*1000} tokens/s total_tokens_count:{input_tokens_count + generated_tokens_count} request_id:{final_output.request_id}  input_tokens_count:{input_tokens_count} generated_tokens_count:{generated_tokens_count}",flush=True)    
    
    INFER_TOKEN_METRICS.inc(f"infer_{INFERENCE_NAME}_input_tokens_num",input_tokens_count,tags={"request_id":final_output.request_id})
    INFER_TOKEN_METRICS.inc(f"infer_{INFERENCE_NAME}_output_tokens_num", generated_tokens_count,tags={"request_id":final_output.request_id})
    INFER_TOKEN_METRICS.push()
    
    return [(generated_text,{"metadata":{
        "request_id":final_output.request_id,
        "input_tokens_count":input_tokens_count,
        "generated_tokens_count":generated_tokens_count,
        "time_cost":time_cost,
        "first_token_time":first_token_time-current_time_milliseconds,
        "speed":float(generated_tokens_count)/time_cost*1000,
        "prob":prob
    }})]   

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

    if infer_mode == "ray/vllm":        
        num_gpus = int(sys_conf.get("num_gpus",1))
        print(f"infer_mode:{infer_mode} tensor_parallel_size: {num_gpus}")
        global INFERENCE_NAME
        INFERENCE_NAME = infer_params.get("udfName","auto")

        try:
            ray.get_actor("VLLM_STREAM_SERVER")
        except ValueError:            
            ray.remote(VLLMStreamServer).options(name="VLLM_STREAM_SERVER",lifetime="detached",max_concurrency=1000).remote()
        
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
        
        disable_log_stats: bool = infer_params.get("backend.disable_log_stats","false") == "true"
        quantization = infer_params.get("backend.quantization",None)
        

        ohter_params = {}
        if quantization is not None:
            ohter_params["quantization"] = quantization

        if "backend.max_num_batched_tokens" in infer_params:
            ohter_params["max_num_batched_tokens"] = int(infer_params["backend.max_num_batched_tokens"])

        if "backend.max_model_len" in infer_params:
            ohter_params["max_model_len"] = int(infer_params["backend.max_model_len"])

        if "backend.max_num_seqs" in infer_params:
            ohter_params["max_num_seqs"] = int(infer_params["backend.max_num_seqs"])        
        

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
            disable_log_stats=disable_log_stats,            
            ** ohter_params
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)                               
        llm.async_stream_chat = types.MethodType(async_vllm_chat, llm) 
        llm.async_get_meta = types.MethodType(async_get_meta,llm)
        return (llm,llm.engine.tokenizer)  

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

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,trust_remote_code=True)    

    quatization = infer_params.get("quatization", "false")

    if quatization in ["4", "8", "true"]:
        print(f"enable [{quatization}] quatization.", flush=True)
        load_in_8bit = quatization == "8"
        # default using int4
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if load_in_8bit:
            llm_int8_threshold = infer_params.get("llm_int8_threshold", 6.0)
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=llm_int8_threshold,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=False,
                llm_int8_has_fp16_weight=False,
            )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_dir,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
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

    def get_meta(self): 
        config = self.config   
        return [{
            "model_deploy_type": "proprietary",
            "backend":"transformers",
            "max_model_len":getattr(config, "model_max_length", -1),
            "architectures":getattr(config, "architectures", [])
        }]    

    model.stream_chat = types.MethodType(stream_chat, model)
    model.get_meta = types.MethodType(get_meta, model)     
    return (model,tokenizer)


def sft_train(data_refs:List[DataServer],
              train_params:Dict[str,str],
              conf: Dict[str, str])->Generator[BlockRow,Any,Any]:
    from ..utils.sft import sft_train as common_sft_train
    return common_sft_train(data_refs,train_params,conf) 

