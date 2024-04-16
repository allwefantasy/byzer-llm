import os
from os.path import expanduser
import urllib.request
import tarfile
from loguru import logger

class StorageSubCommand:
        
    @staticmethod
    def install(args):
        version = args.version
        home = expanduser("~")
        auto_coder_dir = os.path.join(home, ".auto-coder")
        if not os.path.exists(auto_coder_dir):
            os.makedirs(auto_coder_dir)

        logger.info(f"Download Byzer Retrieval version {version}")
        download_url = f"https://download.byzer.org/byzer-retrieval/byzer-retrieval-lib-{version}.tar.gz"
        libs_dir = os.path.join(auto_coder_dir, "storage", "libs")
        os.makedirs(libs_dir, exist_ok=True)
        download_path = os.path.join(libs_dir, f"byzer-retrieval-lib-{version}.tar.gz")
        
        def download_with_progressbar(url, filename):
            def progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\rDownload progress: {percent}%", end="")
        
            urllib.request.urlretrieve(url, filename,reporthook=progress)
        
        download_with_progressbar(download_url, download_path)    

        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=libs_dir)
        
        print("Byzer Retrieval installed successfully")
    
    @staticmethod
    def start(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval
        version = args.version
        home = expanduser("~")
        auto_coder_dir = os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(auto_coder_dir, "storage", "libs", f"byzer-retrieval-lib-{version}")
        data_dir = os.path.join(auto_coder_dir, "storage", "data","cluster_0")

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if not os.path.exists(libs_dir):
            StorageSubCommand.install(args)

        code_search_path = [libs_dir]
        
        logger.info(f"Connect and start Byzer Retrieval version {version}")
        env_vars = byzerllm.connect_cluster(address=args.ray_address,code_search_path=code_search_path)
        
        retrieval = ByzerRetrieval()
        retrieval.launch_gateway()

        builder = retrieval.cluster_builder()
        builder.set_name("cluster_0").set_location(data_dir).set_num_nodes(1).set_node_cpu(1).set_node_memory("2g")
        builder.set_java_home(env_vars["JAVA_HOME"]).set_path(env_vars["PATH"]).set_enable_zgc()
        builder.start_cluster()
        
        print(retrieval.cluster_info("cluster_0"))
        print("Byzer Retrieval started successfully")