import argparse
import os
import ray
import shlex
import jinja2
import yaml
from byzerllm.utils.client import ByzerLLM, InferBackend
from byzerllm.apps.command import StorageSubCommand

# 命令和参数的中英文映射字典
locales = {
    "desc": {
        "en": "Byzer-LLM command line tool",
        "zh": "Byzer-LLM 命令行工具"
    },
    "help_deploy": {
        "en": "Deploy a model",
        "zh": "部署一个模型"
    },
    "help_ray_address": {
        "en": "Ray cluster address",
        "zh": "Ray 集群地址"
    },
    "help_num_workers": {
        "en": "Number of model workers",
        "zh": "模型工作节点数"
    },
    "help_gpus_per_worker": {
        "en": "Number of GPUs per worker",
        "zh": "每个工作节点的 GPU 数"
    },
     "help_cpus_per_worker": {
        "en": "Number of CPUs per worker",
        "zh": "每个工作节点的 CPU 数"
    },
    "help_model_path": {
        "en": "Local model directory path",
        "zh": "本地模型目录路径"
    },
    "help_pretrained_model_type": {
        "en": "Pretrained model type",
        "zh": "预训练模型类型"
    },
    "help_udf_name": {
        "en": "Deployed model name",
        "zh": "部署后的模型名称"
    },
    "help_infer_params": {
        "en": "Model inference parameters",
        "zh": "模型推理参数"
    },
    "help_infer_backend": {
        "en": "Model inferrence Backend",
        "zh": "模型推理后端"
    },
    "help_query": {
        "en": "Query a deployed model",
        "zh": "查询一个已部署的模型"
    },
    "help_query_model": {
        "en": "Deployed model UDF name",
        "zh": "已部署的模型 UDF 名称"
    },
    "help_query_text": {
        "en": "User query/prompt",
        "zh": "用户查询/提示"
    },
    "help_template": {
        "en": "Chat template",
        "zh": "对话模板"
    },
    "deploy_success": {
        "en": "Model {0} deployed successfully",
        "zh": "模型 {0} 部署成功"
    },
    "undeploy_success": {
        "en": "Model {0} undeployed successfully",
        "zh": "模型 {0} 卸载成功"
    },
    "already_deployed": {
        "en": "Model {0} already deployed",
        "zh": "模型 {0} 已经部署过了"
    }
}

# 获取系统语言环境
lang = os.getenv("LANG", "en").split(".")[0]
if lang.startswith("zh"):
    lang = "zh"
else:
    lang = "en"

class StoreNestedDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = {}
        for kv in values:
            try:
                for clean_kv in shlex.split(kv):
                    key, value =clean_kv.split("=", 1)
                    d[key]=value
            except ValueError:
                raise argparse.ArgumentError(self, f"Invalid argument format: {kv}")
        setattr(namespace, self.dest, d)  

def main():
    parser = argparse.ArgumentParser(description=locales["desc"][lang])
    subparsers = parser.add_subparsers(dest='command')

    # Deploy 子命令
    deploy_cmd = subparsers.add_parser('deploy', help=locales["help_deploy"][lang])
    deploy_cmd.add_argument('--ray_address', default="auto", help=locales["help_ray_address"][lang])
    deploy_cmd.add_argument('--num_workers', type=int, default=1, help=locales["help_num_workers"][lang])
    deploy_cmd.add_argument('--gpus_per_worker', type=float, default=0, help=locales["help_gpus_per_worker"][lang])
    deploy_cmd.add_argument('--cpus_per_worker', type=float, default=0.01, help=locales["help_cpus_per_worker"][lang])
    deploy_cmd.add_argument('--model_path', default="",required=False, help=locales["help_model_path"][lang])
    deploy_cmd.add_argument('--pretrained_model_type', default="custom/llama2", help=locales["help_pretrained_model_type"][lang])
    deploy_cmd.add_argument('--model', required=True, help=locales["help_udf_name"][lang])    
    deploy_cmd.add_argument('--infer_backend', default="vllm", help=locales["help_infer_backend"][lang])
    deploy_cmd.add_argument('--infer_params', nargs='+', action=StoreNestedDict, help=locales["help_infer_params"][lang])
    deploy_cmd.add_argument("--file", default=None, required=False, help="")
    
    # Undeploy 子命令
    deploy_cmd = subparsers.add_parser('undeploy', help=locales["help_deploy"][lang])
    deploy_cmd.add_argument('--ray_address', default="auto", help=locales["help_ray_address"][lang])
    deploy_cmd.add_argument('--model', required=True, help=locales["help_udf_name"][lang])    
    deploy_cmd.add_argument("--file", default=None, required=False, help="")

    # Query 子命令
    query_cmd = subparsers.add_parser('query', help=locales["help_query"][lang])
    query_cmd.add_argument('--ray_address', default="auto", help=locales["help_ray_address"][lang])
    query_cmd.add_argument('--model', required=True, help=locales["help_query_model"][lang])
    query_cmd.add_argument('--query', required=True, help=locales["help_query_text"][lang])
    query_cmd.add_argument('--template', default="auto", help=locales["help_template"][lang])
    query_cmd.add_argument("--file", default=None, required=False, help="")

    # Storage 子命令
    storage_cmd = subparsers.add_parser('storage', help='Manage Byzer Storage')    
    storage_cmd_subparsers = storage_cmd.add_subparsers(dest='storage_command')
    
    # install 子命令
    storage_install_cmd = storage_cmd_subparsers.add_parser('install', help='Install Byzer Storage')
    storage_install_cmd.add_argument("--file", default=None, required=False, help="")
    storage_install_cmd.add_argument('--ray_address', default="auto", help=locales["help_ray_address"][lang])
    storage_install_cmd.add_argument("--version", default="0.1.11", required=False, help="")
    storage_install_cmd.add_argument('--cluster', default="byzerai_store", help="")
    storage_install_cmd.add_argument('--base_dir', default="", help="")
    
    # start 子命令
    storage_start_command = storage_cmd_subparsers.add_parser('start', help='Start Byzer Storage')
    storage_start_command.add_argument("--file", default=None, required=False, help="")
    storage_start_command.add_argument("--version", default="0.1.11", required=False, help="")
    storage_start_command.add_argument('--ray_address', default="auto", help=locales["help_ray_address"][lang])
    storage_start_command.add_argument('--cluster', default="byzerai_store", help="")
    storage_start_command.add_argument('--base_dir', default="", help="")


    # stop 子命令
    storage_stop_command = storage_cmd_subparsers.add_parser('stop', help='Stop Byzer Storage')
    storage_stop_command.add_argument("--file", default=None, required=False, help="")
    storage_stop_command.add_argument("--version", default="0.1.11", required=False, help="")
    storage_stop_command.add_argument('--ray_address', default="auto", help=locales["help_ray_address"][lang])
    storage_stop_command.add_argument('--cluster', default="byzerai_store", help="")
    storage_stop_command.add_argument('--base_dir', default="", help="")
    
    # export 子命令
    storage_export_command = storage_cmd_subparsers.add_parser('export', help='Export Byzer Storage Information')    
    storage_export_command.add_argument("--file", default=None, required=False, help="")
    storage_export_command.add_argument("--version", default="0.1.11", required=False, help="")
    storage_export_command.add_argument('--ray_address', default="auto", help=locales["help_ray_address"][lang])
    storage_export_command.add_argument('--cluster', default="byzerai_store", help="")
    storage_export_command.add_argument('--base_dir', default="", help="")
    

    args = parser.parse_args()
    
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
        print(f"{arg:20}: {value}")
    print("-" * 50)    

    if args.command == 'deploy':
        ray.init(address=args.ray_address, namespace="default", ignore_reinit_error=True)

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
        ray.init(address=args.ray_address, namespace="default", ignore_reinit_error=True)

        llm_client = ByzerLLM()
        llm_client.setup_template(args.model, args.template)

        resp = llm_client.chat_oai(model=args.model, conversations=[{
            "role": "user",
            "content": args.query,
        }])
        print(resp[0].output,flush=True)

    elif args.command == 'undeploy':
        ray.init(address=args.ray_address, namespace="default", ignore_reinit_error=True)

        llm = ByzerLLM()
        llm.undeploy(args.model)

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
            

        

if __name__ == "__main__":
    main()