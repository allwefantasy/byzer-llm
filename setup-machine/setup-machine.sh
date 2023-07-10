set -e # Exit immediately if any command exits with a non-zero status (i.e., an error).
set -x # Print each command before executing it, prefixed with the + character. This can be useful for debugging shell scripts.
set -u # Treat unset variables as an error and exit immediately.
set -o pipefail # # Exit with a non-zero status if any command in a pipeline fails, rather than just the last command.

echo ""
echo ""



ROLE=${ROLE:-"master"}
OS="ubuntu"
BYZER_VERSION="2.3.8"
BYZER_NOTEBOOK_VERSION="1.2.5"

cat <<EOF
This script will help you install Byzer-LLM enviroment on your machine (CentOS or Ubuntu)

1. Make sure the script is executed by root user.
2. Byzer-lang Version: ${BYZER_VERSION}
3. Byzer-notebook Version: ${BYZER_NOTEBOOK_VERSION}
EOF

# check USER_PASSWORD is set or not
if [[ -z "${USER_PASSWORD}" ]]; then
    echo "We will try to create a user byzerllm  in this Machine. You should specify the USER_PASSWORD of byzerllm first"
    echo "The new password of byzerllm is: "
    read USER_PASSWORD    
fi

USER_PASSWORD=${USER_PASSWORD:-""}


# Check if the system is running CentOS
if [ -f /etc/centos-release ]; then
    OS="centos"
fi

# Check if the system is running Ubuntu
if [ -f /etc/lsb-release ]; then
    . /etc/lsb-release
    if [ "$DISTRIB_ID" == "Ubuntu" ]; then
        OS="ubuntu"
    fi
fi

echo "current system is ${OS}"


if command -v git &> /dev/null; then
    echo "git is installed"    
else
    echo "git is not installed, now install git"    
    if [ "$OS" = "ubuntu" ]; then
        apt install -y git
    elif [ "$OS" = "centos" ]; then
        yum install -y git
    fi
fi


echo "Setup basic user byzerllm "
groupadd ai
useradd -m byzerllm -g ai
echo "byzerllm:${USER_PASSWORD}" | sudo chpasswd

echo "Setup sudo permission for byzerllm"
echo "byzerllm ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    
echo "swith to user byzerllm"
su - byzerllm

echo "Install Conda environment"

CONDA_INSTALL_PATH=$HOME/miniconda3

echo "Download the latest version of Miniconda"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
chmod +x ~/miniconda.sh
./miniconda.sh -b -p $HOME/miniconda3

# echo "export PATH=\"$CONDA_INSTALL_PATH/bin:\$PATH\"" >> ~/.bashrc
echo "Initialize conda and activate the base environment"
"$CONDA_INSTALL_PATH/bin/conda" init bash
source ~/.bashrc

echo "Now check the nvidia driver and toolkit"

TOOLKIT_INSTALLED=true
DRIVER_INSTALLED=true

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver are installed"    
else
    echo "NVIDIA driver are not installed"    
    DRIVER_INSTALLED=false
fi


if [[ $DRIVER_INSTALLED == false ]];then
    echo "Now install the NVIDIA driver"
    if [ "$OS" = "ubuntu" ]; then
        sudo apt install -y  nvidia-driver-535
    elif [ "$OS" = "centos" ]; then
        sudo dnf config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
        # here we install cuda also, but it will not be used since will use the cuda installed by conda
        sudo dnf install -y cuda 
    fi
fi

echo "Now install the NVIDIA toolkit with conda"
conda install -y cuda==11.7.0 -c nvidia


if command -v nvcc &> /dev/null; then
    echo "NVIDIA toolkit is installed from conda"
else
    echo "Fail to install NVIDIA toolkit from conda"
    exit 1
fi


echo "Create conda environments: byzerllm-dev"
conda create -y --name byzerllm-dev python=3.10.11
conda activate byzerllm-dev

echo "Create some basic folders: models projects byzerllm_stroage softwares data"

mkdir models projects byzerllm_stroage softwares data

echo "Install some basic python packages"
git clone https://gitee.com/allwefantasy/byzer-llm
pip install -r byzer-llm/demo-requirements.txt

echo "Setup TGI support in Byzer-LLM"

git clone https://gitee.com/mirrors/text-generation-inference
cd  text-generation-inference/server/custom_kernels
pip install .

cd byzer-llm

echo "Install tgi flash attention dependency, it may take a while"
make install-flash-attention

echo "Install TGI vllm dependency it may take a while "
make install-vllm


cd $HOME/softwares

if [[ $ROLE == "master" ]];then
    echo "Since we are in master mode, we should install Byzer Lang and Byzer Notebook"

    wget "https://download.byzer.org/byzer/byzer-lang/${BYZER_VERSION}/byzer-lang-all-in-one-linux-amd64-3.3.0-2.3.8.tar.gz" -O byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}.tar.gz
    tar -zxvf byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}.tar.gz


    wget "https://download.byzer.org/byzer/byzer-notebook/${BYZER_NOTEBOOK_VERSION}/byzer-notebook-${BYZER_NOTEBOOK_VERSION}.tar.gz" -O byzer-notebook-${BYZER_NOTEBOOK_VERSION}.tar.gz
    tar -zxvf byzer-notebook-${BYZER_NOTEBOOK_VERSION}.tar.gz

    echo "Setup JDK"

    cat <<EOF >> ~/.bashrc
export JAVA_HOME=$HOME/softwares/byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}"/jdk8
export PATH=${JAVA_HOME}/bin:$PATH
EOF

    source activate ~/.bashrc

    echo "Setup MySQL"
    if command -v docker &> /dev/null; then
        echo "docker is installed"    
    else
        echo "docker is not installed, now install docker"    
        if [ "$OS" = "ubuntu" ]; then
            sudo apt install -y docker.io
        elif [ "$OS" = "centos" ]; then
            sudo yum install -y docker.io
        fi
    fi

    echo "Start MySQL"
    docker run --name metadb -e MYSQL_ROOT_PASSWORD=mlsql -p 3306:3306 -d mysql:5.7

    cat <<EOF >> ~/softwares/ray.start.master.sh
ray stop && ray start --head --dashboard-host 127.0.0.1  '--resources={"master": 1, "passwordless_ssh_node": 1000}'
EOF    

    cat <<EOF
1. The byzer-lang is installed at $HOME/softwares/byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}
   1.1 Use `./conf/byzer.properties.override` to config byzer-lang
   1.2 Use `./bin/byzer.sh start` to start byzer-lang

2. Byzer Notebook deepends on MySQL 5.7, you can use the following command to install MySQL 5.7
   docker run --name metadb -e MYSQL_ROOT_PASSWORD=mlsql -p 3306:3306 -d mysql:5.7

3. The byzer-notebook is installed at $HOME/softwares/byzer-notebook
   3.1 Use `./conf/notebook.properties` to config byzer-lang
   3.2 Use `./bin/notebook.sh start` to start byzer-lang

4. ray start script is installed at $HOME/softwares/ray.start.master.sh
   4.1 You can use `bash ray.start.master.sh` to start ray cluster
   4.2 You can use `bash ray.start.worker.sh` to start ray worker

5. Please according to the https://docs.byzer.org/#/byzer-lang/zh-cn/byzer-llm/deploy to setup the byzer-lang and byzer-notebook
EOF

else

    cat <<EOF >> ~/softwares/ray.start.worker.sh
echo "The master ip is: "
read masterIP
echo "The name of this worker is: "
read workerName
ray stop && ray start --address="${masterIP}:6379"  "--resources={\"${workerName}\": 1}"
EOF  

    cat <<EOF
ray start script is installed at $HOME/softwares/ray.start.master.sh
    You can use `bash ray.start.master.sh` to start ray cluster
    You can use `bash ray.start.worker.sh` to start ray worker
EOF    

fi