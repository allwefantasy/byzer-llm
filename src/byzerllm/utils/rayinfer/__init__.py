import ray
from ray import serve
from aviary.backend.server.run import llm_server,LLMApp 
from aviary.backend.server.models import *
import ray._private.usage.usage_lib
from typing import Union
import socket

def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _build_static_app(
        udfName:str,
        model_dir:str,
        initializer:Annotated[Union[DeepSpeed, DeviceMap, SingleDevice, LlamaCpp],Field(discriminator="type"),],
        num_workers:int=1,
        **kwargs        
):            
    model_id = udfName
    llmapp = LLMApp(
        deployment_config= DeploymentConfig(
            max_concurrent_queries=64,
            ray_actor_options= {"resources": {"master": 0.0001}},
        ),
        scaling_config=ScalingConfig(num_workers=num_workers, 
                                       num_gpus_per_worker=kwargs.get("num_gpus_per_worker",1), 
                                       num_cpus_per_worker=1, 
                                       pg_timeout_s=6000),
        model_config=StaticBatchingModel(
          batching="static",
          model_id=model_id,
          model_url=model_dir,
          max_input_words=800,
          initialization=StaticBatchingInitializationConfig(
            hf_model_id=model_dir,
            initializer=initializer,
            pipeline="transformers",
          ),
          generation=StaticBatchingGenerationConfig(
            max_input_words=800,
            max_batch_size=18,
            generate_kwargs={"do_sample":True,
                             "max_new_tokens":512,
                             "min_new_tokens":16,
                             "temperature":0.7,
                             "repetition_penalty":1.1,
                             "top_p":0.8,"top_k":5,
                             "return_token_type_ids":False},
            prompt_format={"system": "{{instruction}}\\n","assistant": "{{instruction}}\\n",
                           "trailing_assistant": "{{instruction}}\\n",
                           "user": "{{instruction}}\\n",
                           "default_system_message": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
            stopping_sequences=[],)
            )
          )       
    return llmapp
    
   
          

def build_model_serving(udfName,model_dir,mode,num_workers:int=1):
    """Run the LLM Server on the local Ray Cluster

    Args:
        *models: A list of LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/")           # run all models in the models directory
       run("models/model.yaml") # run one model in the model directory
       run({...LLMApp})         # run a single LLMApp
       run("models/model1.yaml", "models/model2.yaml", {...LLMApp}) # mix and match
    """
    initializer = DeviceMap(dtype="bfloat16", from_pretrained_kwargs={"trust_remote_code": True, "use_cache": True}, 
                                  use_bettertransformer=False, torch_compile=TorchCompile(backend="inductor", mode="max-autotune"))
    if mode == "deepspeed":
       initializer = DeepSpeed(dtype="float16", from_pretrained_kwargs={"trust_remote_code": True}, use_kernel=True, max_tokens=1536)  
    
    model_yaml = _build_static_app(udfName,model_dir,initializer,num_workers=num_workers)

    router, deployments, deployment_routes, app_names = llm_server([model_yaml])
    ray._private.usage.usage_lib.record_library_usage("aviary")
    model_id = [item for item in deployments.keys()][0]
    app = deployments[model_id]
    route = deployment_routes[model_id]
    app_name = app_names[model_id]

    available_port = _get_free_port()
    model_infer = serve.run(
            app,
            name=app_name,
            route_prefix=route,
            host="127.0.0.1",
            port=available_port,
            _blocking=False,
        )
    print(f"[{model_id}] [{app_name}] Model serving is running on 127.0.0.1:{available_port}",flush=True)
    return model_infer
        