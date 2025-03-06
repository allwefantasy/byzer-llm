import argparse
import shlex
from byzerllm.lang import lang, locales
from typing import List, Optional


class StoreNestedDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = {}
        for kv in values:
            try:
                for clean_kv in shlex.split(kv):
                    key, value = clean_kv.split("=", 1)
                    d[key] = value
            except ValueError:
                raise argparse.ArgumentError(self, f"Invalid argument format: {kv}")
        setattr(namespace, self.dest, d)


def get_command_args(input_args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=locales["desc"][lang])
    subparsers = parser.add_subparsers(dest="command")

    # Deploy 子命令
    deploy_cmd = subparsers.add_parser("deploy", help=locales["help_deploy"][lang])
    deploy_cmd.add_argument(
        "--ray_address", default="auto", help=locales["help_ray_address"][lang]
    )
    deploy_cmd.add_argument(
        "--num_workers", type=int, default=1, help=locales["help_num_workers"][lang]
    )
    deploy_cmd.add_argument(
        "--worker_concurrency",
        type=int,
        default=1,
        help=locales["help_worker_concurrency"][lang],
    )
    deploy_cmd.add_argument(
        "--gpus_per_worker",
        type=float,
        default=0,
        help=locales["help_gpus_per_worker"][lang],
    )
    deploy_cmd.add_argument(
        "--cpus_per_worker",
        type=float,
        default=0.01,
        help=locales["help_cpus_per_worker"][lang],
    )
    deploy_cmd.add_argument(
        "--model_path",
        default="",
        required=False,
        help=locales["help_model_path"][lang],
    )
    deploy_cmd.add_argument(
        "--pretrained_model_type",
        default="custom/llama2",
        help=locales["help_pretrained_model_type"][lang],
    )
    deploy_cmd.add_argument(
        "--model", required=True, help=locales["help_udf_name"][lang]
    )
    deploy_cmd.add_argument(
        "--infer_backend", default="", help=locales["help_infer_backend"][lang]
    )
    deploy_cmd.add_argument(
        "--infer_params",
        nargs="+",
        action=StoreNestedDict,
        help=locales["help_infer_params"][lang],
    )
    deploy_cmd.add_argument("--file", default=None, required=False, help="")

    # Undeploy 子命令
    deploy_cmd = subparsers.add_parser("undeploy", help=locales["help_deploy"][lang])
    deploy_cmd.add_argument(
        "--ray_address", default="auto", help=locales["help_ray_address"][lang]
    )
    deploy_cmd.add_argument(
        "--model", required=True, help=locales["help_udf_name"][lang]
    )
    deploy_cmd.add_argument("--file", default=None, required=False, help="")
    deploy_cmd.add_argument("--force", action="store_true", help="")

    # Query 子命令
    query_cmd = subparsers.add_parser("query", help=locales["help_query"][lang])
    query_cmd.add_argument(
        "--ray_address", default="auto", help=locales["help_ray_address"][lang]
    )
    query_cmd.add_argument(
        "--model", required=True, help=locales["help_query_model"][lang]
    )
    query_cmd.add_argument(
        "--query", required=True, help=locales["help_query_text"][lang]
    )
    query_cmd.add_argument(
        "--template", default="auto", help=locales["help_template"][lang]
    )
    query_cmd.add_argument("--file", default=None, required=False, help="")
    query_cmd.add_argument("--output_file", default="", help="")

    # Stat Command
    status_cmd = subparsers.add_parser("stat", help=locales["help_query"][lang])
    status_cmd.add_argument(
        "--ray_address", default="auto", help=locales["help_ray_address"][lang]
    )
    status_cmd.add_argument(
        "--model", required=True, help=locales["help_query_model"][lang]
    )
    status_cmd.add_argument("--file", default=None, required=False, help="")
    status_cmd.add_argument(
        "--interval", type=int, default=0, help="Interval in seconds for continuous stat printing"
    )
    

    # Storage 子命令
    storage_cmd = subparsers.add_parser("storage", help="Manage Byzer Storage")
    storage_cmd_subparsers = storage_cmd.add_subparsers(dest="storage_command")

    emb_parser = storage_cmd_subparsers.add_parser("emb", help="Manage embedding model")
    emb_subparsers = emb_parser.add_subparsers(dest="emb_command", required=True)
    emb_start_parser = emb_subparsers.add_parser("start", help="Start embedding model")
    emb_stop_parser = emb_subparsers.add_parser("stop", help="Stop embedding model")

    model_memory_parser = storage_cmd_subparsers.add_parser(
        "model_memory", help="Manage long-term memory model"
    )
    model_memory_subparsers = model_memory_parser.add_subparsers(
        dest="model_memory_command", required=True
    )
    model_memory_start_parser = model_memory_subparsers.add_parser(
        "start", help="Start long-term memory model"
    )
    model_memory_stop_parser = model_memory_subparsers.add_parser(
        "stop", help="Stop long-term memory model"
    )

    # Add common arguments for all new subcommands
    for _subparser in [
        emb_start_parser,
        emb_stop_parser,
        model_memory_start_parser,
        model_memory_stop_parser,
    ]:
        _subparser.add_argument("--file", default=None, required=False, help="")
        _subparser.add_argument(
            "--ray_address", default="auto", help=locales["help_ray_address"][lang]
        )
        _subparser.add_argument("--version", default="0.1.11", required=False, help="")
        _subparser.add_argument("--cluster", default="byzerai_store", help="")
        _subparser.add_argument("--base_dir", default="", help="")

    # install 子命令
    storage_install_cmd = storage_cmd_subparsers.add_parser(
        "install", help="Install Byzer Storage"
    )
    storage_install_cmd.add_argument("--file", default=None, required=False, help="")
    storage_install_cmd.add_argument(
        "--ray_address", default="auto", help=locales["help_ray_address"][lang]
    )
    storage_install_cmd.add_argument(
        "--version", default="0.1.11", required=False, help=""
    )
    storage_install_cmd.add_argument("--cluster", default="byzerai_store", help="")
    storage_install_cmd.add_argument("--base_dir", default="", help="")

    storage_collection_cmd = storage_cmd_subparsers.add_parser("collection", help="")
    storage_collection_cmd.add_argument("--file", default=None, required=False, help="")
    storage_collection_cmd.add_argument(
        "--ray_address", default="auto", help=locales["help_ray_address"][lang]
    )
    storage_collection_cmd.add_argument(
        "--version", default="0.1.11", required=False, help=""
    )
    storage_collection_cmd.add_argument("--cluster", default="byzerai_store", help="")
    storage_collection_cmd.add_argument("--base_dir", default="", help="")
    storage_collection_cmd.add_argument("--name", default="", help="")
    storage_collection_cmd.add_argument("--description", default="", help="")

    # start 子命令
    storage_start_command = storage_cmd_subparsers.add_parser(
        "start", help="Start Byzer Storage"
    )
    storage_start_command.add_argument("--file", default=None, required=False, help="")
    storage_start_command.add_argument(
        "--version", default="0.1.11", required=False, help=""
    )
    storage_start_command.add_argument(
        "--ray_address", default="auto", help=locales["help_ray_address"][lang]
    )
    storage_start_command.add_argument("--cluster", default="byzerai_store", help="")
    storage_start_command.add_argument("--base_dir", default="", help="")
    storage_start_command.add_argument(
        "--enable_emb", action="store_true", help="Enable embedding model"
    )    
    storage_start_command.add_argument(
        "--enable_model_memory",
        action="store_true",
        help="Enable model memory",
    )
    storage_start_command.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes in the cluster",
    )
    storage_start_command.add_argument(
        "--node_cpus",
        type=int,
        default=1,
        help="Number of CPUs per node",
    )
    storage_start_command.add_argument(
        "--node_memory",
        type=int,
        default=2,
        help="Memory per node in GB",
    )

    # stop 子命令
    storage_stop_command = storage_cmd_subparsers.add_parser(
        "stop", help="Stop Byzer Storage"
    )
    storage_stop_command.add_argument("--file", default=None, required=False, help="")
    storage_stop_command.add_argument(
        "--version", default="0.1.11", required=False, help=""
    )
    storage_stop_command.add_argument(
        "--ray_address", default="auto", help=locales["help_ray_address"][lang]
    )
    storage_stop_command.add_argument("--cluster", default="byzerai_store", help="")
    storage_stop_command.add_argument("--base_dir", default="", help="")

    # export 子命令
    storage_export_command = storage_cmd_subparsers.add_parser(
        "export", help="Export Byzer Storage Information"
    )
    storage_export_command.add_argument("--file", default=None, required=False, help="")
    storage_export_command.add_argument(
        "--version", default="0.1.11", required=False, help=""
    )
    storage_export_command.add_argument(
        "--ray_address", default="auto", help=locales["help_ray_address"][lang]
    )
    storage_export_command.add_argument("--cluster", default="byzerai_store", help="")
    storage_export_command.add_argument("--base_dir", default="", help="")

    # Serve 子命令
    serve_cmd = subparsers.add_parser(
        "serve", help="Serve deployed models with OpenAI compatible APIs"
    )
    serve_cmd.add_argument("--file", default="", help="")
    serve_cmd.add_argument("--ray_address", default="auto", help="")
    serve_cmd.add_argument("--host", default="", help="")
    serve_cmd.add_argument("--port", type=int, default=8000, help="")
    serve_cmd.add_argument("--uvicorn_log_level", default="info", help="")
    serve_cmd.add_argument("--allow_credentials", action="store_true", help="")
    serve_cmd.add_argument("--allowed_origins", default=["*"], help="")
    serve_cmd.add_argument("--allowed_methods", default=["*"], help="")
    serve_cmd.add_argument("--allowed_headers", default=["*"], help="")
    serve_cmd.add_argument("--api_key", default="", help="")
    serve_cmd.add_argument("--served_model_name", default="", help="")
    serve_cmd.add_argument("--prompt_template", default="", help="")
    serve_cmd.add_argument("--ssl_keyfile", default="", help="")
    serve_cmd.add_argument("--ssl_certfile", default="", help="")
    serve_cmd.add_argument("--response_role", default="assistant", help="")
    serve_cmd.add_argument(
        "--template", default="auto", help=locales["help_template"][lang]
    )

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)
    return args
