import os
from os.path import expanduser
import subprocess
import urllib.request
import tarfile
import ray

class StorageSubCommand:
        
    @staticmethod
    def install(args):
        home = expanduser("~")
        auto_coder_dir = os.path.join(home, ".auto-coder")
        if not os.path.exists(auto_coder_dir):
            os.makedirs(auto_coder_dir)
        
        java_version = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
        if "21" not in str(java_version):
            raise Exception("JDK 21 is required")

        download_url = "https://download.byzer.org/byzer-retrieval/byzer-retrieval-lib-0.1.11.tar.gz"
        libs_dir = os.path.join(auto_coder_dir, "storage", "libs")
        os.makedirs(libs_dir, exist_ok=True)
        download_path = os.path.join(libs_dir, "byzer-retrieval-lib-0.1.11.tar.gz")

        urllib.request.urlretrieve(download_url, download_path)

        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=libs_dir)
        
        print("Byzer Retrieval installed successfully")
    
    @staticmethod
    def start(args):
        home = expanduser("~")
        auto_coder_dir = os.path.join(home, ".auto-coder")
        libs_dir = os.path.join(auto_coder_dir, "storage", "libs", "byzer-retrieval-lib-0.1.11")
        
        code_search_path = [libs_dir]
        java_home = subprocess.check_output(["which", "java"], universal_newlines=True).strip()
        java_home = java_home[:-9]  # remove /bin/java from the path
        path = java_home + "/bin:" + os.environ["PATH"]
        env_vars = {"JAVA_HOME": java_home, "PATH": path}
        
        ray.init(address="auto", namespace="default", 
                 job_config=ray.job_config.JobConfig(code_search_path=code_search_path,
                                                     runtime_env={"env_vars": env_vars}))

        from byzerllm.utils.retrieval import ByzerRetrieval

        retrieval = ByzerRetrieval()
        retrieval.launch_gateway()

        builder = retrieval.cluster_builder()
        builder.set_name("cluster1").set_location("/tmp/cluster1").set_num_nodes(2).set_node_cpu(1).set_node_memory("3g")
        builder.set_java_home(env_vars["JAVA_HOME"]).set_path(env_vars["PATH"]).set_enable_zgc()
        builder.start_cluster()

        print("Byzer Retrieval started successfully")