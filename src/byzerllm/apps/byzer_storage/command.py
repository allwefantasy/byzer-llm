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
    def start(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval

        version = args.version
        cluster = args.cluster
        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")

        libs_dir = os.path.join(
            base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}"
        )
        data_dir = os.path.join(base_dir, "storage", "data")

        console.print("[bold green]Starting Byzer Storage...[/bold green]")

        if not os.path.exists(os.path.join(data_dir, cluster)):
            os.makedirs(data_dir, exist_ok=True)
            rprint("[green]✓[/green] Created data directory")

        if not os.path.exists(libs_dir):
            with console.status("[bold blue]Installing Byzer Storage...[/bold blue]"):
                StorageSubCommand.install(args)
            rprint("[green]✓[/green] Installed Byzer Storage")

        code_search_path = [libs_dir]

        with console.status("[bold blue]Connecting to cluster...[/bold blue]"):
            env_vars = byzerllm.connect_cluster(
                address=args.ray_address, code_search_path=code_search_path
            )
        rprint("[green]✓[/green] Connected to cluster")

        with console.status("[bold blue]Launching gateway...[/bold blue]"):
            retrieval = ByzerRetrieval()
            retrieval.launch_gateway()
        rprint("[green]✓[/green] Launched gateway")

        if retrieval.is_cluster_exists(name=cluster):
            console.print(
                Panel(
                    f"[yellow]Cluster {cluster} already exists. Please stop it first.[/yellow]"
                )
            )
            return

        base_model_dir = os.path.join(base_dir, "storage", "models")
        os.makedirs(base_model_dir, exist_ok=True)
        bge_model = os.path.join(base_model_dir, "AI-ModelScope", "bge-large-zh")

        with console.status("[bold blue]Checking GPU availability...[/bold blue]"):
            has_gpu = torch.cuda.is_available()
        if has_gpu:
            rprint("[green]✓[/green] GPU detected")
        else:
            rprint("[yellow]![/yellow] No GPU detected, using CPU")

        with console.status("[bold blue]Checking embedding model...[/bold blue]"):
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
                    rprint(
                        f"[yellow]![/yellow] and place it in the directory: {bge_model}"
                    )
                    rprint("[yellow]![/yellow] Then restart this process.")
            else:
                model_path = bge_model
                rprint(f"[green]✓[/green] Embedding model found: {model_path}")

        llm = byzerllm.ByzerLLM()
        if args.enable_emb and downloaded and not llm.is_model_exist("emb"):
            with console.status("[bold blue]Deploying embedding model...[/bold blue]"):
                llm.setup_num_workers(1).setup_infer_backend(InferBackend.Transformers)
                if has_gpu:
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
            rprint("[green]✓[/green] Deployed embedding model")

        if args.enable_model_memory and has_gpu:
            with console.status("[bold blue]Checking Long-Memory model...[/bold blue]"):
                downloaded = True
                llama_model = os.path.join(
                    base_model_dir, "meta-llama", "Meta-Llama-3-8B-Instruct-GPTQ"
                )
                if not os.path.exists(llama_model):
                    downloaded = False
                    try:
                        model_path = snapshot_download(
                            model_id="meta-llama/Meta-Llama-3-8B-Instruct-GPTQ",
                            cache_dir=base_model_dir,
                            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                        )
                        rprint(
                            f"[green]✓[/green] Long-Memory model downloaded: {model_path}"
                        )
                        downloaded = True
                    except Exception as e:
                        rprint(
                            f"[red]✗[/red] Failed to download Long-Memory model: {str(e)}"
                        )
                        rprint(
                            "[yellow]![/yellow] Please manually download the model 'meta-llama/Meta-Llama-3-8B-Instruct-GPTQ'"
                        )
                        rprint(
                            f"[yellow]![/yellow] and place it in the directory: {llama_model}"
                        )
                        rprint("[yellow]![/yellow] Then restart this process.")
                else:
                    rprint("[green]✓[/green] Long-Memory model already exists")

            if downloaded:
                with console.status("[bold blue]Checking dependencies...[/bold blue]"):
                    check_dependencies()
                with console.status(
                    "[bold blue]Starting long-term memory model...[/bold blue]"
                ):
                    llm.setup_gpus_per_worker(1).setup_cpus_per_worker(
                        0.001
                    ).setup_num_workers(1)
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
                        rprint("[green]✓[/green] Long-term memory model started")
                    except Exception as e:
                        rprint(
                            f"[red]✗[/red] Failed to start Long-Memory model: {str(e)}"
                        )
                        rprint(
                            "[yellow]![/yellow] Please check the error message and try again."
                        )

        cluster_json = os.path.join(base_dir, "storage", "data", f"{cluster}.json")
        if os.path.exists(cluster_json):
            with console.status("[bold blue]Restoring Byzer Storage...[/bold blue]"):
                StorageSubCommand.restore(args)
            console.print(
                Panel("[green]Byzer Storage restored and started successfully[/green]")
            )
            return

        with console.status("[bold blue]Starting cluster...[/bold blue]"):
            builder = retrieval.cluster_builder()
            builder.set_name(cluster).set_location(data_dir).set_num_nodes(
                1
            ).set_node_cpu(1).set_node_memory("2g")
            builder.set_java_home(env_vars["JAVA_HOME"]).set_path(
                env_vars["PATH"]
            ).set_enable_zgc()
            builder.start_cluster()

        with open(
            os.path.join(base_dir, "storage", "data", f"{cluster}.json"), "w"
        ) as f:
            f.write(json.dumps(retrieval.cluster_info(cluster), ensure_ascii=False))

        console.print(Panel("[green]Byzer Storage started successfully[/green]"))

    @staticmethod
    def stop(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval

        version = args.version
        cluster = args.cluster
        home = expanduser("~")
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")

        error_summary = []

        with console.status("[bold red]Stopping Byzer Storage...") as status:
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

            status.update("[bold blue]Connecting to cluster...")
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

            status.update("[bold blue]Launching gateway...")
            try:
                retrieval = ByzerRetrieval()
                retrieval.launch_gateway()
                rprint("[green]✓[/green] Gateway launched")
            except Exception as e:
                rprint(f"[red]✗[/red] Failed to launch gateway: {str(e)}")
                error_summary.append(
                    "Failed to launch gateway. Please check if Byzer Retrieval is properly installed."
                )

            status.update(f"[bold blue]Shutting down cluster {cluster}...")
            try:
                retrieval.shutdown_cluster(cluster_name=cluster)
                rprint(f"[green]✓[/green] Cluster {cluster} shut down")
            except Exception as e:
                rprint(f"[red]✗[/red] Failed to shut down cluster {cluster}: {str(e)}")
                error_summary.append(
                    f"Failed to shut down cluster {cluster}. You may need to manually stop it."
                )

            llm = byzerllm.ByzerLLM()

            status.update("[bold blue]Undeploying embedding model...")
            try:
                llm.undeploy("emb")
                rprint("[green]✓[/green] Embedding model undeployed")
            except Exception as e:
                rprint(f"[red]✗[/red] Failed to undeploy embedding model: {str(e)}")
                error_summary.append(
                    "Failed to undeploy embedding model. You may need to manually undeploy it."
                )

            status.update("[bold blue]Undeploying long-term memory model...")
            try:
                llm.undeploy("long_memory")
                rprint("[green]✓[/green] Long-term memory model undeployed")
            except Exception as e:
                rprint(
                    f"[red]✗[/red] Failed to undeploy long-term memory model: {str(e)}"
                )
                error_summary.append(
                    "Failed to undeploy long-term memory model. You may need to manually undeploy it."
                )

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

        with console.status("[bold blue]Exporting Byzer Storage...") as status:
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

            status.update("[bold blue]Connecting to cluster...")
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

            status.update("[bold blue]Launching gateway...")
            try:
                retrieval = ByzerRetrieval()
                retrieval.launch_gateway()
                rprint("[green]✓[/green] Gateway launched")
            except Exception as e:
                rprint(f"[red]✗[/red] Failed to launch gateway: {str(e)}")
                error_summary.append(
                    "Failed to launch gateway. Please check if Byzer Retrieval is properly installed."
                )

            status.update(f"[bold blue]Exporting cluster {cluster} information...")
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

        with console.status("[bold blue]Restoring Byzer Storage...") as status:
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

            status.update("[bold blue]Connecting to cluster...")
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

            status.update("[bold blue]Launching gateway...")
            try:
                retrieval = ByzerRetrieval()
                retrieval.launch_gateway()
                rprint("[green]✓[/green] Gateway launched")
            except Exception as e:
                rprint(f"[red]✗[/red] Failed to launch gateway: {str(e)}")
                error_summary.append(
                    "Failed to launch gateway. Please check if Byzer Retrieval is properly installed."
                )

            status.update(f"[bold blue]Restoring cluster {cluster}...")
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
