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
import torch
from modelscope.hub.snapshot_download import snapshot_download
from byzerllm.utils.client import InferBackend
import huggingface_hub
import byzerllm
from pydantic import BaseModel
from huggingface_hub import snapshot_download as hf_snapshot_download


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
    def download_model(model_id: str, cache_dir: str, local_files_only: bool = False) -> str:
        console.print(f"[bold blue]Downloading model {model_id}...")
        try:
            if "meta-llama" in model_id:
                model_path = hf_snapshot_download(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only
                )
            else:
                model_path = snapshot_download(
                    model_id=model_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only
                )
            console.print(f"[green]✓[/green] Model downloaded: {model_path}")
            return model_path
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to download model: {str(e)}")
            console.print(f"[yellow]![/yellow] Please manually download the model '{model_id}'")
            console.print(f"[yellow]![/yellow] and place it in the directory: {cache_dir}")
            return None

    @staticmethod
    def emb_start(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval

        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        base_model_dir = os.path.join(base_dir, "storage", "models")
        bge_model = os.path.join(base_model_dir, "AI-ModelScope", "bge-large-zh")

        console.print("[bold blue]Starting embedding model...")

        if not os.path.exists(bge_model):
            bge_model = StorageSubCommand.download_model("AI-ModelScope/bge-large-zh", base_model_dir, local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE)
            if not bge_model:
                return

        byzerllm.connect_cluster(address=args.ray_address)
        llm = byzerllm.ByzerLLM()

        if llm.is_model_exist("emb"):
            console.print("[yellow]![/yellow] Embedding model already deployed.")
            return

        console.print("[bold blue]Deploying embedding model...")
        llm.setup_num_workers(1).setup_infer_backend(InferBackend.Transformers)

        if torch.cuda.is_available():
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

    @staticmethod
    def emb_stop(args):
        import byzerllm

        console.print("[bold blue]Stopping embedding model...")
        byzerllm.connect_cluster(address=args.ray_address)
        llm = byzerllm.ByzerLLM()

        try:
            llm.undeploy("emb")
            console.print("[green]✓[/green] Embedding model stopped successfully")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to stop embedding model: {str(e)}")

    @staticmethod
    def model_memory_start(args):
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
            llama_model = StorageSubCommand.download_model("meta-llama/Meta-Llama-3-8B-Instruct-GPTQ", base_model_dir)
            if not llama_model:
                return

        if not torch.cuda.is_available():
            console.print(
                "[red]✗[/red] GPU not available. Long-term memory model requires GPU."
            )
            return

        byzerllm.connect_cluster(address=args.ray_address)
        llm = byzerllm.ByzerLLM()

        if llm.is_model_exist("long_memory"):
            console.print("[yellow]![/yellow] Long-term memory model already deployed.")
            return

        console.print("[bold blue]Checking dependencies...")
        check_dependencies()

        console.print("[bold blue]Deploying long-term memory model...")
        llm.setup_gpus_per_worker(0.9).setup_cpus_per_worker(0.001).setup_num_workers(1)
        llm.setup_infer_backend(InferBackend.VLLM)
        try:
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
            console.print(
                f"[red]✗[/red] Failed to deploy long-term memory model: {str(e)}"
            )

    @staticmethod
    def model_memory_stop(args):
        import byzerllm

        console.print("[bold blue]Stopping long-term memory model...")
        byzerllm.connect_cluster(address=args.ray_address)
        llm = byzerllm.ByzerLLM()

        try:
            llm.undeploy("long_memory")
            console.print(
                "[green]✓[/green] Long-term memory model stopped successfully"
            )
        except Exception as e:
            console.print(
                f"[red]✗[/red] Failed to stop long-term memory model: {str(e)}"
            )

    @staticmethod
    def install(args):
        version = args.version
        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(
            base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}"
        )
        if os.path.exists(libs_dir):
            print(f"Byzer Storage version {version} already installed.")
            return

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
                logger.error(
                    f"Error downloading and installing JDK 21: {str(e)}. You may need to install JDK 21 manually."
                )

        download_url = f"https://download.byzer.org/byzer-retrieval/byzer-retrieval-lib-{version}.tar.gz"
        libs_dir = os.path.join(base_dir, "storage", "libs")

        os.makedirs(libs_dir, exist_ok=True)
        download_path = os.path.join(libs_dir, f"byzer-retrieval-lib-{version}.tar.gz")
        if os.path.exists(download_path):
            logger.info(f"Byzer Storage version {version} already downloaded.")
        else:

            def download_with_progressbar(url, filename):
                def progress(count, block_size, total_size):
                    percent = int(count * block_size * 100 / total_size)
                    print(f"\rDownload progress: {percent}%", end="")

                urllib.request.urlretrieve(url, filename, reporthook=progress)

            logger.info(f"Download Byzer Storage version {version}: {download_url}")
            download_with_progressbar(download_url, download_path)

            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(path=libs_dir)

        print("Byzer Storage installed successfully")

    def collection(args):
        from byzerllm.apps.llama_index.collection_manager import (
            CollectionManager,
            CollectionItem,
        )

        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        collection_manager = CollectionManager(base_dir)
        if args.name:
            collection = CollectionItem(name=args.name, description=args.description)
            collection_manager.add_collection(collection)
            print(f"Collection {args.name} added successfully.")
        else:
            print("Please provide collection name.")

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
        store_location = StorageSubCommand.get_store_location(args)

        console.print("[bold green]Starting Byzer Storage...[/bold green]")

        if not os.path.exists(
            os.path.join(store_location.data_dir, store_location.cluster)
        ):
            os.makedirs(store_location.data_dir, exist_ok=True)
            rprint("[green]✓[/green] Created data directory")

        if not os.path.exists(store_location.libs_dir):
            rprint("[bold blue]Installing Byzer Storage...[/bold blue]")
            StorageSubCommand.install(args)
            rprint("[green]✓[/green] Installed Byzer Storage")

        code_search_path = [store_location.libs_dir]

        console.print("[bold blue]Connecting to cluster...[/bold blue]")
        env_vars = byzerllm.connect_cluster(
            address=args.ray_address, code_search_path=code_search_path
        )
        rprint("[green]✓[/green] Connected to cluster")
        return (env_vars, store_location)

    @staticmethod
    def start(args):
        from byzerllm.utils.retrieval import ByzerRetrieval

        env_vars, store_location = StorageSubCommand.connect_cluster(args)

        console.print("[bold blue]Launching gateway...[/bold blue]")
        retrieval = ByzerRetrieval()
        retrieval.launch_gateway()
        rprint("[green]✓[/green] Launched gateway")

        if retrieval.is_cluster_exists(name=store_location.cluster):
            console.print(
                Panel(
                    f"[yellow]Cluster {store_location.cluster} already exists. Please stop it first.[/yellow]"
                )
            )
            return

        base_model_dir = os.path.join(store_location.base_dir, "storage", "models")
        os.makedirs(base_model_dir, exist_ok=True)
        bge_model = os.path.join(base_model_dir, "AI-ModelScope", "bge-large-zh")

        console.print("[bold blue]Checking GPU availability...[/bold blue]")
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            rprint("[green]✓[/green] GPU detected")
        else:
            rprint("[yellow]![/yellow] No GPU detected, using CPU")

        console.print("[bold blue]Checking embedding model...[/bold blue]")
        downloaded = True
        if not os.path.exists(bge_model):
            downloaded = False
            try:
                model_path = snapshot_download(
                    model_id="AI-ModelScope/bge-large-zh",
                    cache_dir=base_model_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                )
                rprint(f"[green]✓[/green] Embedding model downloaded: {model_path}")
                downloaded = True
            except Exception as e:
                rprint(f"[red]✗[/red] Failed to download embedding model: {str(e)}")
                rprint(
                    "[yellow]![/yellow] Please manually download the model 'AI-ModelScope/bge-large-zh'"
                )
                rprint(f"[yellow]![/yellow] and place it in the directory: {bge_model}")
                rprint("[yellow]![/yellow] Then restart this process.")
        else:
            model_path = bge_model
            rprint(f"[green]✓[/green] Embedding model found: {model_path}")

        if args.enable_emb and downloaded:
            StorageSubCommand.emb_start(args)

        if args.enable_model_memory and has_gpu:
            StorageSubCommand.model_memory_start(args)

        cluster_json = os.path.join(
            store_location.base_dir, "storage", "data", f"{store_location.cluster}.json"
        )
        if os.path.exists(cluster_json):
            console.print("[bold blue]Restoring Byzer Storage...[/bold blue]")
            StorageSubCommand.restore(args)
            console.print(
                Panel("[green]Byzer Storage restored and started successfully[/green]")
            )
            return

        console.print("[bold blue]Starting cluster...[/bold blue]")
        builder = retrieval.cluster_builder()
        builder.set_name(store_location.cluster).set_location(
            store_location.data_dir
        ).set_num_nodes(1).set_node_cpu(1).set_node_memory("2g")
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
                    retrieval.cluster_info(store_location.cluster), ensure_ascii=False
                )
            )

        console.print(Panel("[green]Byzer Storage started successfully[/green]"))

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
            console.print("[red]✗[/red] No instance found.")
            error_summary.append(
                "No instance found. Please check if Byzer Storage is properly installed."
            )
            return

        env_vars, _ = StorageSubCommand.connect_cluster(args)
        console.print("[bold blue]Launching gateway...")
        try:
            retrieval = ByzerRetrieval()
            retrieval.launch_gateway()
            rprint("[green]✓[/green] Gateway launched")
        except Exception as e:
            rprint(f"[red]✗[/red] Failed to launch gateway: {str(e)}")
            error_summary.append(
                "Failed to launch gateway. Please check if Byzer Retrieval is properly installed."
            )

        console.print(f"[bold blue]Shutting down cluster {store_location.cluster}...")
        try:
            retrieval.shutdown_cluster(cluster_name=store_location.cluster)
            rprint(f"[green]✓[/green] Cluster {store_location.cluster} shut down")
        except Exception as e:
            rprint(
                f"[red]✗[/red] Failed to shut down cluster {store_location.cluster}: {str(e)}"
            )
            error_summary.append(
                f"Failed to shut down cluster {store_location.cluster}. You may need to manually stop it."
            )

        StorageSubCommand.emb_stop(args)
        StorageSubCommand.model_memory_stop(args)

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

    @staticmethod
    def export(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval

        version = args.version
        cluster = args.cluster
        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")

        error_summary = []

        console.print("[bold blue]Exporting Byzer Storage...")
        libs_dir = os.path.join(
            base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}"
        )
        cluster_json = os.path.join(base_dir, "storage", "data", f"{cluster}.json")

        if not os.path.exists(cluster_json) or not os.path.exists(libs_dir):
            console.print("[red]✗[/red] No instance found.")
            error_summary.append(
                "No instance found. Please check if Byzer Storage is properly installed."
            )
            return

        code_search_path = [libs_dir]

        console.print("[bold blue]Connecting to cluster...")
        try:
            byzerllm.connect_cluster(
                address=args.ray_address, code_search_path=code_search_path
            )
            rprint("[green]✓[/green] Connected to cluster")
        except Exception as e:
            rprint(f"[red]✗[/red] Failed to connect to cluster: {str(e)}")
            error_summary.append(
                "Failed to connect to cluster. Please check your network connection and Ray setup."
            )

        console.print("[bold blue]Launching gateway...")
        try:
            retrieval = ByzerRetrieval()
            retrieval.launch_gateway()
            rprint("[green]✓[/green] Gateway launched")
        except Exception as e:
            rprint(f"[red]✗[/red] Failed to launch gateway: {str(e)}")
            error_summary.append(
                "Failed to launch gateway. Please check if Byzer Retrieval is properly installed."
            )

        console.print(f"[bold blue]Exporting cluster {cluster} information...")
        try:
            cluster_info = retrieval.cluster_info(cluster)
            with open(cluster_json, "w") as f:
                json.dump(cluster_info, f, ensure_ascii=False, indent=2)
            rprint(
                f"[green]✓[/green] Cluster {cluster} information exported to {cluster_json}"
            )
        except Exception as e:
            rprint(
                f"[red]✗[/red] Failed to export cluster {cluster} information: {str(e)}"
            )
            error_summary.append(
                f"Failed to export cluster {cluster} information. You may need to check the cluster status."
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

    @staticmethod
    def restore(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval

        version = args.version
        cluster = args.cluster
        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")

        error_summary = []

        console.print("[bold blue]Restoring Byzer Storage...")
        libs_dir = os.path.join(
            base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}"
        )
        cluster_json = os.path.join(base_dir, "storage", "data", f"{cluster}.json")

        if not os.path.exists(cluster_json) or not os.path.exists(libs_dir):
            console.print("[red]✗[/red] No instance found.")
            error_summary.append(
                "No instance found. Please check if Byzer Storage is properly installed."
            )
            return

        code_search_path = [libs_dir]

        console.print("[bold blue]Connecting to cluster...")
        try:
            byzerllm.connect_cluster(
                address=args.ray_address, code_search_path=code_search_path
            )
            rprint("[green]✓[/green] Connected to cluster")
        except Exception as e:
            rprint(f"[red]✗[/red] Failed to connect to cluster: {str(e)}")
            error_summary.append(
                "Failed to connect to cluster. Please check your network connection and Ray setup."
            )

        console.print("[bold blue]Launching gateway...")
        try:
            retrieval = ByzerRetrieval()
            retrieval.launch_gateway()
            rprint("[green]✓[/green] Gateway launched")
        except Exception as e:
            rprint(f"[red]✗[/red] Failed to launch gateway: {str(e)}")
            error_summary.append(
                "Failed to launch gateway. Please check if Byzer Retrieval is properly installed."
            )

        console.print(f"[bold blue]Restoring cluster {cluster}...")
        try:
            if not retrieval.is_cluster_exists(cluster):
                with open(cluster_json, "r") as f:
                    cluster_info = json.load(f)
                retrieval.restore_from_cluster_info(cluster_info)
                rprint(f"[green]✓[/green] Cluster {cluster} restored successfully")
            else:
                rprint(f"[yellow]![/yellow] Cluster {cluster} already exists")
                error_summary.append(
                    f"Cluster {cluster} already exists. No restoration needed."
                )
        except Exception as e:
            rprint(f"[red]✗[/red] Failed to restore cluster {cluster}: {str(e)}")
            error_summary.append(
                f"Failed to restore cluster {cluster}. You may need to check the cluster status and configuration."
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
