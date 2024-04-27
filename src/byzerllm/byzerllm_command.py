import argparse
import os
import ray
import shlex
import jinja2
import yaml
from byzerllm.utils.client import ByzerLLM, InferBackend
from byzerllm.utils.client.types import Templates
from byzerllm.apps.command import StorageSubCommand
from byzerllm.utils.client.entrypoints.openai.serve import serve, ServerArgs
import byzerllm
from byzerllm.lang import locales, lang 
from byzerllm.command_args import StoreNestedDict, get_command_args

def main():
    args = get_command_args()

    if args.file:
        with open(args.file, "r") as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if key != "file":  # 排除 --file 参数本身   
                    ## key: ENV {{VARIABLE_NAME}}
                    if isinstance(value, str) and value.startswith("ENV"):  
                        template = jinja2.Template(value.removeprefix("ENV").strip())                  
                        value = template.render(os.environ)                        
                    setattr(args, key, value)
    
    print("Command Line Arguments:")
    print("-" * 50)
    for arg, value in vars(args).items():        
        if arg == "infer_params" and isinstance(value, dict) and "saas.api_key" in value:  
            import copy
            new_value = copy.deepcopy(value)
            new_value["saas.api_key"] = "******"
            print(f"{arg:20}: {new_value}")
        else:     
            print(f"{arg:20}: {value}")
    print("-" * 50)    

    if args.command == 'deploy':
        byzerllm.connect_cluster(address=args.ray_address)
        llm = ByzerLLM()
        if llm.is_model_exist(args.model):
            print(locales["already_deployed"][lang].format(args.model))
            return
        
        llm.setup_gpus_per_worker(args.gpus_per_worker).setup_cpus_per_worker(args.cpus_per_worker).setup_num_workers(args.num_workers)
        if not args.pretrained_model_type.startswith("saas"):
            if args.infer_backend == "vllm":
                llm.setup_infer_backend(InferBackend.VLLM)
            elif args.infer_backend == "llama_cpp": 
                llm.setup_infer_backend(InferBackend.LLAMA_CPP)
            elif args.infer_backend == "transformers":
                llm.setup_infer_backend(InferBackend.Transformers)
            elif args.infer_backend == "deepspeed":    
                llm.setup_infer_backend(InferBackend.DeepSpeed)
            else:
                raise ValueError("Invalid infer_backend")
        
        llm.deploy(model_path=args.model_path,
                pretrained_model_type=args.pretrained_model_type,
                udf_name=args.model,
                infer_params=args.infer_params or {})

        print(locales["deploy_success"][lang].format(args.model))

    elif args.command == 'query':
        byzerllm.connect_cluster(address=args.ray_address)

        llm_client = ByzerLLM()        
        if args.template == "default":
            llm_client.setup_template(args.model, template=Templates.default())
        elif args.template == "llama":
            llm_client.setup_template(args.model, template=Templates.llama())    
        elif args.template == "qwen":
            llm_client.setup_template(args.model, template=Templates.qwen())        
        elif args.template == "yi":
            llm_client.setup_template(args.model, template=Templates.yi())  
        elif args.template == "empty":
            llm_client.setup_template(args.model, template=Templates.empty())          
        else:
            llm_client.setup_template(args.model, args.template)

        resp = llm_client.chat_oai(model=args.model, conversations=[{
            "role": "user",
            "content": args.query,
        }])
        
        if args.output_file:
            ext = os.path.splitext(args.output_file)[1]
            if ext == ".mp3" or ext == ".wav":
                import base64
                with open(args.output_file, "wb") as f:
                    f.write(base64.b64decode(resp[0].output))
            else:        
                with open(args.output_file, "w") as f:
                    f.write(resp[0].output)
        else:        
            print(resp[0].output,flush=True)

    elif args.command == 'undeploy':
        byzerllm.connect_cluster(address=args.ray_address)

        llm = ByzerLLM()
        llm.undeploy(args.model,force=args.force)

        print(locales["undeploy_success"][lang].format(args.model))
        
    elif args.command == 'storage':
        if args.storage_command == "install":
            StorageSubCommand.install(args)
        elif args.storage_command == "start":
            StorageSubCommand.start(args) 
        elif args.storage_command == "stop":
            StorageSubCommand.stop(args)
        elif args.storage_command == "export":
            StorageSubCommand.export(args)
    
    elif args.command == 'serve':
        byzerllm.connect_cluster(address=args.ray_address) 
        llm_client = ByzerLLM()
        if args.served_model_name:
            llm_client.setup_template(args.served_model_name, args.template)                 
        server_args = ServerArgs(**{arg: getattr(args, arg) for arg in vars(ServerArgs())})                      
        serve(llm=llm_client, args=server_args)            