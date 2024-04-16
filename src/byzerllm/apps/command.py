import os
from os.path import expanduser
import urllib.request
import tarfile
from loguru import logger

import os
import re
from packaging import version

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
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        logger.info(f"Download Byzer Retrieval version {version}")
        download_url = f"https://download.byzer.org/byzer-retrieval/byzer-retrieval-lib-{version}.tar.gz"
        libs_dir = os.path.join(base_dir, "storage", "libs")
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
        database = args.database
        home = expanduser("~")        
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}")
        data_dir = os.path.join(base_dir, "storage", "data",database)

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
        builder.set_name(database).set_location(data_dir).set_num_nodes(1).set_node_cpu(1).set_node_memory("2g")
        builder.set_java_home(env_vars["JAVA_HOME"]).set_path(env_vars["PATH"]).set_enable_zgc()
        builder.start_cluster()
        
        print(retrieval.cluster_info(database))
        print("Byzer Retrieval started successfully")

    @staticmethod 
    def stop(args):    
        import byzerllm
        from byzerllm.utils.retrieval import ByzerRetrieval
        version = args.version
        database = args.database
        home = expanduser("~")        
        base_dir = args.base_dir or os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(base_dir, "storage", "libs", f"byzer-retrieval-lib-{version}")
        data_dir = os.path.join(base_dir, "storage", "data",database)        

        if not os.path.exists(data_dir) or not os.path.exists(libs_dir):
            print("No instance find.")
            return

        code_search_path = [libs_dir]
        
        logger.info(f"Connect and start Byzer Retrieval version {version}")
        byzerllm.connect_cluster(address=args.ray_address,code_search_path=code_search_path)
     
        retrieval = ByzerRetrieval()
        retrieval.launch_gateway()
        retrieval.shutdown_cluster(cluster_name=database)
