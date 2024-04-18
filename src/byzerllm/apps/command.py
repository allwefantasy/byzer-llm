import os
from os.path import expanduser
import urllib.request
import tarfile
from loguru import logger

import os
import re
from packaging import version
import json

def get_latest_byzer_retrieval_lib(directory):
    # Define the regex pattern for matching the directories
    pattern = r'^byzer-retrieval-lib-(\d+\.\d+\.\d+)$'
    
    # List all entries in the directory
    entries = os.listdir(directory)
    
    # Initialize an empty list to hold (version, directory name) tuples
    versions = []
    
    # Iterate through the entries and filter directories that match the pattern
    for entry in entries:
        match = re.match(pattern, entry)
        if match:
            # Extract the version part from the directory name
            ver = match.group(1)
            # Append the version and directory name as a tuple
            versions.append((version.parse(ver), entry))
    
    # Sort the list of tuples by the version (first element of the tuple)
    versions.sort()
    
    # Get the last element from the sorted list (the one with the highest version)
    if versions:
        return versions[-1][1]  # Return the directory name of the latest version
    else:
        return None  # Return None if no matching directory is found

class StorageSubCommand:
        
    @staticmethod
    def install(args):
        version = args.version
        home = expanduser("~")        
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}")
        if os.path.exists(libs_dir):
            print(f"Byzer Storage version {version} already installed.")
            return
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir,exist_ok=True)            

        
        download_url = f"https://download.byzer.org/byzer-retrieval/byzer-retrieval-lib-{version}.tar.gz"
        libs_dir = os.path.join(base_dir, "storage", "libs")
        
        os.makedirs(libs_dir, exist_ok=True)
        download_path = os.path.join(libs_dir, f"byzer-retrieval-lib-{version}.tar.gz")
        
        def download_with_progressbar(url, filename):
            def progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\rDownload progress: {percent}%", end="")
        
            urllib.request.urlretrieve(url, filename,reporthook=progress)
        
        logger.info(f"Download Byzer Storage version {version}: {download_url}")
        download_with_progressbar(download_url, download_path)    

        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=libs_dir)
        
        print("Byzer Storage installed successfully")
    
    @staticmethod
    def start(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval
        version = args.version
        cluster = args.cluster
        home = expanduser("~")        
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        
        libs_dir = os.path.join(base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}")
        data_dir = os.path.join(base_dir, "storage", "data")

        if not os.path.exists(os.path.join(data_dir,cluster)):
            os.makedirs(data_dir,exist_ok=True)
                
        if not os.path.exists(libs_dir):            
            StorageSubCommand.install(args)

        code_search_path = [libs_dir]
        
        logger.info(f"Connect and start Byzer Retrieval version {version}")
        env_vars = byzerllm.connect_cluster(address=args.ray_address,code_search_path=code_search_path)
        
        retrieval = ByzerRetrieval()
        retrieval.launch_gateway()

        if retrieval.is_cluster_exists(name=cluster):
            print(f"Cluster {cluster} exists already, stop it first.")
            return 
        
        cluster_json = os.path.join(base_dir, "storage", "data",f"{cluster}.json")
        if os.path.exists(cluster_json):
            StorageSubCommand.restore(args)
            print("Byzer Storage started successfully")
            return 
            
        builder = retrieval.cluster_builder()
        builder.set_name(cluster).set_location(data_dir).set_num_nodes(1).set_node_cpu(1).set_node_memory("2g")
        builder.set_java_home(env_vars["JAVA_HOME"]).set_path(env_vars["PATH"]).set_enable_zgc()
        builder.start_cluster()
        
        with open(os.path.join(base_dir, "storage", "data",f"{cluster}.json"),"w") as f:
            f.write(json.dumps(retrieval.cluster_info(cluster),ensure_ascii=False))

        print("Byzer Storage started successfully")

    @staticmethod 
    def stop(args):    
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval
        version = args.version
        cluster = args.cluster
        home = expanduser("~")        
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}")
        data_dir = os.path.join(base_dir, "storage", "data",cluster)        

        if not os.path.exists(data_dir) or not os.path.exists(libs_dir):
            print("No instance find.")
            return

        code_search_path = [libs_dir]
        
        logger.info(f"Connect and start Byzer Retrieval version {version}")
        byzerllm.connect_cluster(address=args.ray_address,code_search_path=code_search_path)             
        retrieval = ByzerRetrieval()
        retrieval.launch_gateway()
        retrieval.shutdown_cluster(cluster_name=cluster)

    @staticmethod 
    def export(args):   
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval
        version = args.version
        cluster = args.cluster
        home = expanduser("~")        
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}")
        cluster_json = os.path.join(base_dir, "storage", "data",f"{cluster}.json")        

        if not os.path.exists(cluster_json) or not os.path.exists(libs_dir):
            print("No instance find.")
            return

        code_search_path = [libs_dir]
        
        logger.info(f"Connect and restore Byzer Retrieval version {version}")
        byzerllm.connect_cluster(address=args.ray_address,code_search_path=code_search_path)        
     
        retrieval = ByzerRetrieval()
        retrieval.launch_gateway()
        
        with open(cluster_json,"w") as f:
            f.write(json.dumps(retrieval.cluster_info(cluster),ensure_ascii=False))

        print(f"Byzer Storage export successfully. Please check {cluster_json}")    


    
    def restore(args):
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval
        version = args.version
        cluster = args.cluster
        home = expanduser("~")        
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}")
        data_dir = os.path.join(base_dir, "storage", "data",cluster)        

        if not os.path.exists(data_dir) or not os.path.exists(libs_dir):
            print("No instance find.")
            return

        code_search_path = [libs_dir]
        
        logger.info(f"Connect and restore Byzer Retrieval version {version}")
        byzerllm.connect_cluster(address=args.ray_address,code_search_path=code_search_path)        
     
        retrieval = ByzerRetrieval()
        retrieval.launch_gateway()

        if not retrieval.is_cluster_exists(cluster):
            with open(os.path.join(base_dir, "storage", "data",f"{cluster}.json"),"r") as f:
                cluster_info = f.read()
            
            retrieval.restore_from_cluster_info(json.loads(cluster_info))
            
            print("Byzer Storage restore successfully")
        else:
            print(f"Cluster {cluster} is already exists")

