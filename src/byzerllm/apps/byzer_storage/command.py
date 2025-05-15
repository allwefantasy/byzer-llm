import os
from os.path import expanduser
import urllib.request
import tarfile
from loguru import logger
import json
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich import print as rprint

from byzerllm.apps.byzer_storage.env import get_latest_byzer_retrieval_lib
from byzerllm.apps.byzer_storage import env
from modelscope.hub.snapshot_download import snapshot_download
from byzerllm.utils.client import InferBackend
import huggingface_hub
import byzerllm
from pydantic import BaseModel
from huggingface_hub import snapshot_download as hf_snapshot_download
from typing import List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


class StorageLocation(BaseModel):
    version: str
    cluster: str
    base_dir: str
    libs_dir: str
    data_dir: str


console = Console()


def check_dependencies():
    import importlib
    from packaging import version

    dependencies = {"vllm": "0.4.3", "torch": "2.3.0", "flash_attn": "2.5.9.post1"}

    for package, required_version in dependencies.items():
        try:
            module = importlib.import_module(package)
            current_version = module.__version__
            if version.parse(current_version) != version.parse(required_version):
                console.print(
                    f"[yellow]Warning: {package} version mismatch. Required: {required_version}, Found: {current_version}[/yellow]"
                )
        except ImportError:
            console.print(f"[yellow]Warning: {package} is not installed.[/yellow]")


class StorageSubCommand:

    @staticmethod
    def download_model(
        model_id: str,
        cache_dir: str,
        local_files_only: bool = False,
        use_huggingface: bool = False,
    ) -> Tuple[Optional[str], List[str]]:
        error_summary = []
        console.print(f"[bold blue]Downloading model {model_id}...")
        try:
            if use_huggingface:
                model_path = hf_snapshot_download(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            else:
                model_path = snapshot_download(
                    model_id=model_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            console.print(f"[green]✓[/green] Model downloaded: {model_path}")
            return model_path, error_summary
        except Exception as e:
            error_msg = f"Failed to download model: {str(e)}"
            console.print(f"[red]✗[/red] {error_msg}")
            console.print(
                f"[yellow]![/yellow] Please manually download the model '{model_id}'"
            )
            console.print(
                f"[yellow]![/yellow] and place it in the directory: {cache_dir}"
            )
            error_summary.append(error_msg)
            return None, error_summary

    @staticmethod
    def emb_start(args):
        error_summary = []
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval

        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        base_model_dir = os.path.join(base_dir, "storage", "models")
        bge_model = os.path.join(base_model_dir, "AI-ModelScope", "bge-large-zh")

        console.print("[bold blue]Starting embedding model...")

        if not os.path.exists(bge_model):
            bge_model, download_errors = StorageSubCommand.download_model(
                "AI-ModelScope/bge-large-zh",
                base_model_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
            error_summary.extend(download_errors)
            if not bge_model:
                error_summary.append("Failed to download embedding model")
                return error_summary

        try:
            byzerllm.connect_cluster(address=args.ray_address)
            llm = byzerllm.ByzerLLM()

            if llm.is_model_exist("emb"):
                console.print("[yellow]![/yellow] Embedding model already deployed.")
                return error_summary

            console.print("[bold blue]Deploying embedding model...")
            llm.setup_num_workers(1).setup_infer_backend(InferBackend.Transformers)

            if torch and torch.cuda.is_available():
                llm.setup_gpus_per_worker(0.1)
            else:
                llm.setup_gpus_per_worker(0)

            llm.setup_cpus_per_worker(0.01).setup_worker_concurrency(20)
            llm.deploy(
                model_path=bge_model,
                pretrained_model_type="custom/bge",
                udf_name="emb",
                infer_params={},
            )
            console.print("[green]✓[/green] Embedding model deployed successfully")
        except Exception as e:
            error_msg = f"Failed to deploy embedding model: {str(e)}"
            console.print(f"[red]✗[/red] {error_msg}")
            error_summary.append(error_msg)

        return error_summary

    @staticmethod
    def emb_stop(args):
        error_summary = []
        import byzerllm

        console.print("[bold blue]Stopping embedding model...")
        try:
            byzerllm.connect_cluster(address=args.ray_address)
            llm = byzerllm.ByzerLLM()

            llm.undeploy("emb")
            console.print("[green]✓[/green] Embedding model stopped successfully")
        except Exception as e:
            error_msg = f"Failed to stop embedding model: {str(e)}"
            console.print(f"[red]✗[/red] {error_msg}")
            error_summary.append(error_msg)

        return error_summary

    @staticmethod
    def model_memory_start(args):
        error_summary = []
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval

        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        base_model_dir = os.path.join(base_dir, "storage", "models")
        llama_model = os.path.join(
            base_model_dir, "meta-llama", "Meta-Llama-3-8B-Instruct-GPTQ"
        )

        console.print("[bold blue]Starting long-term memory model...")

        if not os.path.exists(llama_model):
            llama_model, download_errors = StorageSubCommand.download_model(
                "meta-llama/Meta-Llama-3-8B-Instruct-GPTQ", base_model_dir
            )
            error_summary.extend(download_errors)
            if not llama_model:
                error_summary.append("Failed to download long-term memory model")
                return error_summary

        if not torch or not torch.cuda.is_available():
            error_msg = "GPU not available. Long-term memory model requires GPU."
            console.print(f"[red]✗[/red] {error_msg}")
            error_summary.append(error_msg)
            return error_summary

        try:
            byzerllm.connect_cluster(address=args.ray_address)
            llm = byzerllm.ByzerLLM()

            if llm.is_model_exist("long_memory"):
                console.print(
                    "[yellow]![/yellow] Long-term memory model already deployed."
                )
                return error_summary

            console.print("[bold blue]Checking dependencies...")
            check_dependencies()

            console.print("[bold blue]Deploying long-term memory model...")
            llm.setup_gpus_per_worker(1).setup_cpus_per_worker(0.001).setup_num_workers(
                1
            )
            llm.setup_infer_backend(InferBackend.VLLM)
            llm.deploy(
                model_path=llama_model,
                pretrained_model_type="custom/auto",
                udf_name="long_memory",
                infer_params={
                    "backend.max_model_len": 8000,
                    "backend.enable_lora": True,
                },
            )
            console.print(
                "[green]✓[/green] Long-term memory model deployed successfully"
            )
        except Exception as e:
            error_msg = f"Failed to deploy long-term memory model: {str(e)}"
            console.print(f"[red]✗[/red] {error_msg}")
            error_summary.append(error_msg)

        return error_summary

    @staticmethod
    def model_memory_stop(args):
        error_summary = []
        import byzerllm

        console.print("[bold blue]Stopping long-term memory model...")
        try:
            byzerllm.connect_cluster(address=args.ray_address)
            llm = byzerllm.ByzerLLM()

            llm.undeploy("long_memory")
            console.print(
                "[green]✓[/green] Long-term memory model stopped successfully"
            )
        except Exception as e:
            error_msg = f"Failed to stop long-term memory model: {str(e)}"
            console.print(f"[red]✗[/red] {error_msg}")
            error_summary.append(error_msg)

        return error_summary

    @staticmethod
    def install(args):
        error_summary = []
        version = args.version
        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(
            base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}"
        )
        if os.path.exists(libs_dir):
            logger.info(f"Byzer Storage version {version} already installed.")
            return error_summary

        try:
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)

            env_info = env.detect_env()

            logger.info("Current environment:")
            logger.info(env_info)

            if env_info.java_version == "" or int(env_info.java_version) < 21:
                logger.info("JDK 21 not found, downloading and installing JDK 21...")
                try:
                    env.download_and_install_jdk21(env_info, base_dir)
                except Exception as e:
                    error_msg = f"Error downloading and installing JDK 21: {str(e)}. You may need to install JDK 21 manually."
                    logger.error(error_msg)
                    error_summary.append(error_msg)
            
            download_url = os.environ.get(
                "BYZER_STORAGE_DOWNLOAD_URL",
                f"https://github.com/allwefantasy/BYZER-RETRIEVAL/releases/download/{version}/byzer-retrieval-lib-{version}.tar.gz",
            )
            libs_dir = os.path.join(base_dir, "storage", "libs")

            os.makedirs(libs_dir, exist_ok=True)
            download_path = os.path.join(
                libs_dir, f"byzer-retrieval-lib-{version}.tar.gz"
            )
            if os.path.exists(download_path):
                logger.info(f"Byzer Storage version {version} already downloaded ({download_path}).")
            else:

                def download_with_progressbar(url, filename):
                    def progress(count, block_size, total_size):
                        percent = int(count * block_size * 100 / total_size)
                        print(f"\rDownload progress: {percent}%", end="")

                    try:
                        urllib.request.urlretrieve(url, filename, reporthook=progress)
                        print()  # New line after progress bar
                        return True
                    except Exception as e:
                        print()  # New line after progress bar
                        logger.error(f"Failed to download from {url}: {str(e)}")
                        logger.info(f"Please manually download from {url} and place it at {filename}")
                        return False
                
                logger.info(f"Downloading Byzer Storage version {version} from: {download_url}")
                if not download_with_progressbar(download_url, download_path):
                    error_summary.append(f"Failed to download Byzer Storage. Please manually download from {download_url} and place it at {download_path}")
                    return error_summary

                if os.path.exists(download_path):
                    try:
                        with tarfile.open(download_path, "r:gz") as tar:
                            tar.extractall(path=libs_dir)
                    except Exception as e:
                        error_msg = f"Failed to extract Byzer Storage: {str(e)}"
                        logger.error(error_msg)
                        error_summary.append(error_msg)

        except Exception as e:
            error_msg = f"Failed to install Byzer Storage: {str(e)}"
            logger.error(error_msg)
            error_summary.append(error_msg)

        return error_summary

    @staticmethod
    def collection(args):
        error_summary = []        
        return error_summary

    @staticmethod
    def get_store_location(args):
        version = args.version
        cluster = args.cluster
        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")

        libs_dir = os.path.join(
            base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}"
        )
        data_dir = os.path.join(base_dir, "storage", "data")
        return StorageLocation(
            version=version,
            cluster=cluster,
            base_dir=base_dir,
            libs_dir=libs_dir,
            data_dir=data_dir,
        )

    @staticmethod
    def connect_cluster(args):
        error_summary = []
        store_location = StorageSubCommand.get_store_location(args)

        console.print("[bold green]Starting Byzer Storage...[/bold green]")

        try:
            if not os.path.exists(
                os.path.join(store_location.data_dir, store_location.cluster)
            ):
                os.makedirs(store_location.data_dir, exist_ok=True)
                rprint("[green]✓[/green] Created data directory")

            if not os.path.exists(store_location.libs_dir):
                rprint("[bold blue]Installing Byzer Storage...[/bold blue]")
                install_errors = StorageSubCommand.install(args)
                error_summary.extend(install_errors)
                if not install_errors:
                    rprint("[green]✓[/green] Installed Byzer Storage")

            code_search_path = [store_location.libs_dir]

            console.print("[bold blue]Connecting to cluster...[/bold blue]")
            env_vars = byzerllm.connect_cluster(
                address=args.ray_address, code_search_path=code_search_path
            )
            rprint("[green]✓[/green] Connected to cluster")
        except Exception as e:
            error_msg = f"Failed to connect to cluster: {str(e)}"
            console.print(f"[red]✗[/red] {error_msg}")
            error_summary.append(error_msg)
            return None, None, error_summary

        return env_vars, store_location, error_summary

    @staticmethod
    def start(args):
        error_summary = []
        from byzerllm.utils.retrieval import ByzerRetrieval

        env_vars, store_location, connect_errors = StorageSubCommand.connect_cluster(
            args
        )
        error_summary.extend(connect_errors)

        if env_vars is None or store_location is None:
            return error_summary

        try:
            console.print("[bold blue]Launching gateway...[/bold blue]")
            retrieval = ByzerRetrieval()
            retrieval.launch_gateway()
            rprint("[green]✓[/green] Launched gateway")

            if retrieval.is_cluster_exists(name=store_location.cluster):
                error_msg = f"Cluster {store_location.cluster} already exists. Please stop it first."
                console.print(Panel(f"[yellow]{error_msg}[/yellow]"))
                error_summary.append(error_msg)
                return error_summary

            base_model_dir = os.path.join(store_location.base_dir, "storage", "models")
            os.makedirs(base_model_dir, exist_ok=True)

            console.print("[bold blue]Checking GPU availability...[/bold blue]")
            has_gpu = torch and torch.cuda.is_available()
            if has_gpu:
                rprint("[green]✓[/green] GPU detected")
            else:
                rprint("[yellow]![/yellow] No GPU detected, using CPU")

            if args.enable_emb:
                emb_errors = StorageSubCommand.emb_start(args)
                error_summary.extend(emb_errors)

            if args.enable_model_memory and has_gpu:
                memory_errors = StorageSubCommand.model_memory_start(args)
                error_summary.extend(memory_errors)

            cluster_json = os.path.join(
                store_location.base_dir,
                "storage",
                "data",
                f"{store_location.cluster}.json",
            )
            if os.path.exists(cluster_json):
                console.print("[bold blue]Restoring Byzer Storage...[/bold blue]")
                restore_errors = StorageSubCommand.restore(args)
                error_summary.extend(restore_errors)
                if not restore_errors:
                    console.print(
                        Panel(
                            "[green]Byzer Storage restored and started successfully[/green]"
                        )
                    )
                return error_summary

            console.print("[bold blue]Starting cluster...[/bold blue]")
            try:
                builder = retrieval.cluster_builder()
                builder.set_name(store_location.cluster).set_location(
                    store_location.data_dir
                ).set_num_nodes(args.num_nodes).set_node_cpu(
                    args.node_cpus
                ).set_node_memory(
                    f"{args.node_memory}g"
                )
                builder.set_java_home(env_vars["JAVA_HOME"]).set_path(
                    env_vars["PATH"]
                ).set_enable_zgc()
                builder.start_cluster()

                with open(
                    os.path.join(
                        store_location.base_dir,
                        "storage",
                        "data",
                        f"{store_location.cluster}.json",
                    ),
                    "w",
                ) as f:
                    f.write(
                        json.dumps(
                            retrieval.cluster_info(store_location.cluster),
                            ensure_ascii=False,
                        )
                    )

                console.print(
                    Panel("[green]Byzer Storage started successfully[/green]")
                )
            except Exception as e:
                error_msg = f"Failed to start Byzer Storage cluster: {str(e)}"
                console.print(f"[red]✗[/red] {error_msg}")
                error_summary.append(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            console.print(f"[red]✗[/red] {error_msg}")
            error_summary.append(error_msg)

        return error_summary

    @staticmethod
    def stop(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval

        error_summary = []
        store_location = StorageSubCommand.get_store_location(args)
        console.print("[bold red]Stopping Byzer Storage...")
        libs_dir = os.path.join(
            store_location.base_dir,
            "storage",
            "libs",
            f"byzer-retrieval-lib-{store_location.version}",
        )
        cluster_json = os.path.join(
            store_location.base_dir, "storage", "data", f"{store_location.cluster}.json"
        )

        if not os.path.exists(cluster_json) or not os.path.exists(libs_dir):
            error_msg = "No instance found. Please check if Byzer Storage is properly installed."
            console.print(f"[red]✗[/red] {error_msg}")
            error_summary.append(error_msg)
            return error_summary

        env_vars, _, connect_errors = StorageSubCommand.connect_cluster(args)
        error_summary.extend(connect_errors)

        if env_vars is None:
            return error_summary

        console.print("[bold blue]Launching gateway...")
        try:
            retrieval = ByzerRetrieval()
            retrieval.launch_gateway()
            rprint("[green]✓[/green] Gateway launched")
        except Exception as e:
            error_msg = f"Failed to launch gateway: {str(e)}"
            rprint(f"[red]✗[/red] {error_msg}")
            error_summary.append(
                f"{error_msg} Please check if Byzer Retrieval is properly installed."
            )

        console.print(f"[bold blue]Shutting down cluster {store_location.cluster}...")
        try:
            retrieval.shutdown_cluster(cluster_name=store_location.cluster)
            rprint(f"[green]✓[/green] Cluster {store_location.cluster} shut down")
        except Exception as e:
            error_msg = (
                f"Failed to shut down cluster {store_location.cluster}: {str(e)}"
            )
            rprint(f"[red]✗[/red] {error_msg}")
            error_summary.append(
                f"{error_msg} You may need to manually stop it."
            )

        emb_stop_errors = StorageSubCommand.emb_stop(args)
        error_summary.extend(emb_stop_errors)

        model_memory_stop_errors = StorageSubCommand.model_memory_stop(args)
        error_summary.extend(model_memory_stop_errors)

        if error_summary:
            console.print(
                Panel("[yellow]Byzer Storage stopped with some issues[/yellow]")
            )
            console.print(
                "[bold red]The following steps need manual attention:[/bold red]"
            )
            for error in error_summary:
                console.print(f"- {error}")
        else:
            console.print(Panel("[green]Byzer Storage stopped successfully[/green]"))

        return error_summary

    @staticmethod
    def export(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval

        error_summary = []
        version = args.version
        cluster = args.cluster
        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")

        console.print("[bold blue]Exporting Byzer Storage...")
        libs_dir = os.path.join(
            base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}"
        )
        cluster_json = os.path.join(base_dir, "storage", "data", f"{cluster}.json")

        if not os.path.exists(cluster_json) or not os.path.exists(libs_dir):
            error_msg = "No instance found. Please check if Byzer Storage is properly installed."
            console.print(f"[red]✗[/red] {error_msg}")
            error_summary.append(error_msg)
            return error_summary

        code_search_path = [libs_dir]

        console.print("[bold blue]Connecting to cluster...")
        try:
            byzerllm.connect_cluster(
                address=args.ray_address, code_search_path=code_search_path
            )
            rprint("[green]✓[/green] Connected to cluster")
        except Exception as e:
            error_msg = f"Failed to connect to cluster: {str(e)}"
            rprint(f"[red]✗[/red] {error_msg}")
            error_summary.append(
                f"{error_msg} Please check your network connection and Ray setup."
            )
            return error_summary

        try:
            retrieval = ByzerRetrieval()
            retrieval.launch_gateway()
            rprint("[green]✓[/green] Gateway launched")
        except Exception as e:
            error_msg = f"Failed to launch gateway: {str(e)}"
            rprint(f"[red]✗[/red] {error_msg}")
            error_summary.append(
                f"{error_msg} Please check if Byzer Retrieval is properly installed."
            )
            return error_summary

        console.print(f"[bold blue]Exporting cluster {cluster} information...")
        try:
            cluster_info = retrieval.cluster_info(cluster)
            with open(cluster_json, "w") as f:
                json.dump(cluster_info, f, ensure_ascii=False, indent=2)
            rprint(
                f"[green]✓[/green] Cluster {cluster} information exported to {cluster_json}"
            )
        except Exception as e:
            error_msg = f"Failed to export cluster {cluster} information: {str(e)}"
            rprint(f"[red]✗[/red] {error_msg}")
            error_summary.append(
                f"{error_msg} You may need to check the cluster status."
            )

        if error_summary:
            console.print(
                Panel("[yellow]Byzer Storage exported with some issues[/yellow]")
            )
            console.print(
                "[bold red]The following steps need manual attention:[/bold red]"
            )
            for error in error_summary:
                console.print(f"- {error}")
        else:
            console.print(Panel("[green]Byzer Storage exported successfully[/green]"))

        return error_summary

    @staticmethod
    def restore(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval

        error_summary = []
        version = args.version
        cluster = args.cluster
        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")

        console.print("[bold blue]Restoring Byzer Storage...")
        libs_dir = os.path.join(
            base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}"
        )
        cluster_json = os.path.join(base_dir, "storage", "data", f"{cluster}.json")

        if not os.path.exists(cluster_json) or not os.path.exists(libs_dir):
            error_msg = "No instance found. Please check if Byzer Storage is properly installed."
            console.print(f"[red]✗[/red] {error_msg}")
            error_summary.append(error_msg)
            return error_summary

        code_search_path = [libs_dir]

        console.print("[bold blue]Connecting to cluster...")
        try:
            byzerllm.connect_cluster(
                address=args.ray_address, code_search_path=code_search_path
            )
            rprint("[green]✓[/green] Connected to cluster")
        except Exception as e:
            error_msg = f"Failed to connect to cluster: {str(e)}"
            rprint(f"[red]✗[/red] {error_msg}")
            error_summary.append(
                f"{error_msg} Please check your network connection and Ray setup."
            )
            return error_summary

        console.print("[bold blue]Launching gateway...")
        try:
            retrieval = ByzerRetrieval()
            retrieval.launch_gateway()
            rprint("[green]✓[/green] Gateway launched")
        except Exception as e:
            error_msg = f"Failed to launch gateway: {str(e)}"
            rprint(f"[red]✗[/red] {error_msg}")
            error_summary.append(
                f"{error_msg} Please check if Byzer Retrieval is properly installed."
            )
            return error_summary

        console.print(f"[bold blue]Restoring cluster {cluster}...")
        try:
            if not retrieval.is_cluster_exists(cluster):
                with open(cluster_json, "r") as f:
                    cluster_info = json.load(f)
                retrieval.restore_from_cluster_info(cluster_info)
                rprint(f"[green]✓[/green] Cluster {cluster} restored successfully")
            else:
                warning_msg = (
                    f"Cluster {cluster} already exists. No restoration needed."
                )
                rprint(f"[yellow]![/yellow] {warning_msg}")
                error_summary.append(warning_msg)
        except Exception as e:
            error_msg = f"Failed to restore cluster {cluster}: {str(e)}"
            rprint(f"[red]✗[/red] {error_msg}")
            error_summary.append(
                f"{error_msg} You may need to check the cluster status and configuration."
            )

        if error_summary:
            console.print(
                Panel("[yellow]Byzer Storage restored with some issues[/yellow]")
            )
            console.print(
                "[bold red]The following steps need manual attention:[/bold red]"
            )
            for error in error_summary:
                console.print(f"- {error}")
        else:
            console.print(Panel("[green]Byzer Storage restored successfully[/green]"))

        return error_summary
