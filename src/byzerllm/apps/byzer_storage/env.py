import sys
import subprocess
import os
import platform
import subprocess
from dataclasses import dataclass
import urllib.request
import tarfile
import zipfile
from loguru import logger
from packaging import version
import re
from os.path import expanduser

def is_apple_m_series():
    # 检查系统是否是 macOS
    if platform.system() != 'Darwin':
        return False

    try:
        # 使用 subprocess 获取系统信息
        result = subprocess.run(['sysctl', 'machdep.cpu.brand_string'], capture_output=True, text=True)
        # 检查输出中是否包含 "Apple M"
        if 'Apple M' in result.stdout:
            return True
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return False

@dataclass
class EnvInfo:
   os_name: str
   os_version: str
   python_version: str
   conda_env: str
   virtualenv: str
   has_bash: bool
   java_home: str
   java_version: str
   cpu: str

def find_jdk21_from_dir(directory):
    
    entries = os.listdir(directory)
    
    # Iterate through the entries and filter directories that match the pattern
    for entry in entries:        
        match = "jdk-21" in entry.lower()
        if match and os.path.isdir(os.path.join(directory, entry)):
            return os.path.join(directory, entry)  # Return the directory name if found
    
    return None  # Return None if no matching directory is found     

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


def detect_env() -> EnvInfo:
    os_name = sys.platform
    os_version = ""
    cpu = "x86_64"
    if os_name == "win32":
        os_version = sys.getwindowsversion().major
    elif os_name == "darwin":
        os_version = subprocess.check_output(["sw_vers", "-productVersion"]).decode('utf-8').strip() 
        if is_apple_m_series():
            cpu = "arm64"
    elif os_name == "linux":
        os_version = subprocess.check_output(["uname", "-r"]).decode('utf-8').strip()
        if "aarch64" in subprocess.check_output(["uname", "-m"]).decode('utf-8').strip().lower():
            cpu = "arm64"

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    conda_env = os.environ.get("CONDA_DEFAULT_ENV")

    virtualenv = os.environ.get("VIRTUAL_ENV")

    has_bash = True
    try:
        subprocess.check_output(["bash", "--version"])
    except:
        has_bash = False 

    default_path = os.path.join(expanduser("~"), ".auto-coder")
    if not os.path.exists(default_path):
        default_path = os.path.join(expanduser("~"), ".byzerllm")
        if not os.path.exists(default_path):
            default_path = ""
    
    java_home = os.environ.get("JAVA_HOME")
    java_version = "0"
    if default_path:
        jdk21_dir = find_jdk21_from_dir(default_path)        
        if jdk21_dir:
            java_home = jdk21_dir
            if os_name == "darwin":
                java_home = os.path.join(java_home, "Contents", "Home")                
            java_version = "21"
    else:                
        if java_home:
            try:
                output = subprocess.check_output([f"{java_home}/bin/java", "-version"], stderr=subprocess.STDOUT, universal_newlines=True)
                version_line = output.splitlines()[0]
                version_match = re.search(r'version "(\d+)', version_line)
                if version_match:
                    java_version = version_match.group(1)
            except (subprocess.CalledProcessError, ValueError) as e:
                logger.warning(f"Error checking Java version: {str(e)}")

    return EnvInfo(
        os_name=os_name,
        os_version=os_version,
        python_version=python_version,
        conda_env=conda_env,
        virtualenv=virtualenv,
        has_bash=has_bash,
        java_version=java_version,
        java_home=java_home,       
        cpu=cpu
    )


def download_with_progressbar(url, filename):
   def progress(count, block_size, total_size):
       percent = int(count * block_size * 100 / total_size)
       print(f"\rDownload progress: {percent}%", end="")

   urllib.request.urlretrieve(url, filename, reporthook=progress)


def download_and_install_jdk21(env_info: EnvInfo, install_dir: str):
   jdk_download_url = ""
   if env_info.os_name == "linux":
       if env_info.cpu == "arm64":
           jdk_download_url = "https://download.oracle.com/java/21/archive/jdk-21.0.2_linux-aarch64_bin.tar.gz"
       else:
           jdk_download_url = "https://download.java.net/java/GA/jdk21.0.2/f2283984656d49d69e91c558476027ac/13/GPL/openjdk-21.0.2_linux-x64_bin.tar.gz"
   elif env_info.os_name == "darwin":
       if env_info.cpu == "arm64": 
           jdk_download_url = "https://download.oracle.com/java/21/archive/jdk-21.0.2_macos-aarch64_bin.tar.gz"
       else:
           jdk_download_url = "https://download.java.net/java/GA/jdk21.0.2/f2283984656d49d69e91c558476027ac/13/GPL/openjdk-21.0.2_macos-x64_bin.tar.gz"
   elif env_info.os_name == "win32":
       jdk_download_url = "https://download.java.net/java/GA/jdk21.0.2/f2283984656d49d69e91c558476027ac/13/GPL/openjdk-21.0.2_windows-x64_bin.zip"

   if jdk_download_url:
       logger.info(f"Downloading JDK 21 from {jdk_download_url}")
       download_path = os.path.join(install_dir, os.path.basename(jdk_download_url))
       download_with_progressbar(jdk_download_url, download_path)

       if env_info.os_name == "win32":
           with zipfile.ZipFile(download_path, "r") as zip_ref:
               zip_ref.extractall(install_dir)
       else:
           with tarfile.open(download_path, "r:gz") as tar:
               tar.extractall(path=install_dir)

       logger.info("JDK 21 downloaded and installed successfully")
   else:
       logger.warning(f"No JDK 21 download URL found for {env_info.os_name}")
