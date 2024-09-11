import os
import ray
import jinja2
import yaml
import subprocess
from typing import List, Optional
from byzerllm.utils.client import ByzerLLM, InferBackend
from byzerllm.utils.client.types import Templates
from byzerllm.apps.byzer_storage.command import StorageSubCommand
from byzerllm.utils.client.entrypoints.openai.serve import serve, ServerArgs
import byzerllm
from byzerllm.lang import locales, lang
from byzerllm.command_args import StoreNestedDict, get_command_args
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
import time
import json

console = Console()


def check_ray_status():
    try:
        subprocess.run(
            ["ray", "status"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def start_ray():
    try:
        subprocess.run(
            ["ray", "start", "--head"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_llm_template(tpl: str):
    if tpl == "default":
        return Templates.default()
    elif tpl == "llama":
        return Templates.llama()
    elif tpl == "qwen":
        return Templates.qwen()
    elif tpl == "yi":
        return Templates.yi()
    elif tpl == "empty":
        return Templates.empty()
    else:
        return tpl


def main(input_args: Optional[List[str]] = None):
    args = get_command_args(input_args=input_args)

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
        if (
            arg == "infer_params"
            and isinstance(value, dict)
            and "saas.api_key" in value
        ):
            import copy

            new_value = copy.deepcopy(value)
            new_value["saas.api_key"] = "******"
            print(f"{arg:20}: {new_value}")
        else:
            print(f"{arg:20}: {value}")
    print("-" * 50)

    if args.ray_address == "auto":
        with console.status("[bold green]Checking Ray status...") as status:
            if check_ray_status():
                console.print("[green]✓[/green] Ray is already running")
            else:
                console.print("[yellow]![/yellow] Ray is not running. Starting Ray...")
                if start_ray():
                    console.print("[green]✓[/green] Ray started successfully")
                else:
                    console.print(
                        "[red]✗[/red] Failed to start Ray. Please start Ray manually."
                    )
                    return

    if args.command == "deploy":
        byzerllm.connect_cluster(address=args.ray_address)
        llm = ByzerLLM()
        if llm.is_model_exist(args.model):
            print(locales["already_deployed"][lang].format(args.model))
            return

        llm.setup_gpus_per_worker(args.gpus_per_worker).setup_cpus_per_worker(
            args.cpus_per_worker
        ).setup_num_workers(args.num_workers)
        llm.setup_worker_concurrency(args.worker_concurrency)
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

        llm.deploy(
            model_path=args.model_path,
            pretrained_model_type=args.pretrained_model_type,
            udf_name=args.model,
            infer_params=args.infer_params or {},
        )

        print(locales["deploy_success"][lang].format(args.model))

    elif args.command == "query":
        byzerllm.connect_cluster(address=args.ray_address)

        llm_client = ByzerLLM()
        llm_client.setup_template(args.model, template=get_llm_template(args.template))

        resp = llm_client.chat_oai(
            model=args.model,
            conversations=[
                {
                    "role": "user",
                    "content": args.query,
                }
            ],
        )

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
            print(resp[0].output, flush=True)

    elif args.command == "undeploy":
        byzerllm.connect_cluster(address=args.ray_address)

        llm = ByzerLLM()
        llm.undeploy(args.model, force=args.force)

        print(locales["undeploy_success"][lang].format(args.model))

    elif args.command == "stat":
        byzerllm.connect_cluster(address=args.ray_address)
        model = ray.get_actor(args.model)    
        def print_stat():                                    
            v = ray.get(model.stat.remote())
            o = JSON(json.dumps(v, ensure_ascii=False))
            console.print(Panel(o, title=f"Model Stat"))

        if args.interval > 0:
            try:
                while True:
                    print_stat()
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStat printing stopped.")
        else:
            print_stat()

    elif args.command == "storage":
        if args.storage_command == "install":
            StorageSubCommand.install(args)
        elif args.storage_command == "start":
            StorageSubCommand.start(args)
        elif args.storage_command == "stop":
            StorageSubCommand.stop(args)
        elif args.storage_command == "export":
            StorageSubCommand.export(args)
        elif args.storage_command == "collection":
            StorageSubCommand.collection(args)
        elif args.storage_command == "emb":
            if args.emb_command == "start":
                StorageSubCommand.emb_start(args)
            elif args.emb_command == "stop":
                StorageSubCommand.emb_stop(args)
        elif args.storage_command == "model_memory":
            if args.model_memory_command == "start":
                StorageSubCommand.model_memory_start(args)
            elif args.model_memory_command == "stop":
                StorageSubCommand.model_memory_stop(args)

    elif args.command == "serve":
        byzerllm.connect_cluster(address=args.ray_address)
        llm_client = ByzerLLM()
        if args.served_model_name:
            llm_client.setup_template(args.served_model_name, get_llm_template(args.template))
        server_args = ServerArgs(
            **{arg: getattr(args, arg) for arg in vars(ServerArgs())}
        )
        serve(llm=llm_client, args=server_args)
