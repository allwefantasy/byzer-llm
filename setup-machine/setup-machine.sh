#! /bin/bash

set -e # Exit immediately if any command exits with a non-zero status (i.e., an error).
# set -x # Print each command before executing it, prefixed with the + character. This can be useful for debugging shell scripts.
set -o pipefail # # Exit with a non-zero status if any command in a pipeline fails, rather than just the last command.

echo ""
echo ""

echo "Welcome to Byzer-LLM setup script"

cd ~

ROLE=${ROLE:-"master"}
OS="ubuntu"
BYZER_VERSION="2.3.8"
BYZER_NOTEBOOK_VERSION="1.2.5"
DEFUALT_MYSQL_PASSWORD=${DEFUALT_MYSQL_PASSWORD:-"mlsql"}
TGI_SUPPORT=${TGI_SUPPORT:-"false"}
VLLM_SUPPORT=${VLLM_SUPPORT:-"false"}
AVIARY_SUPPORT=${AVIARY_SUPPORT:-"false"}
NOTEBOOK_LOGO=${NOTEBOOK_LOGO:-"Byzer Notebook"}

CUDA_DNN_SUPPORT=${CUDA_DNN_SUPPORT:-"false"}
PYPI_MIRROR=${PYPI_MIRROR:-"aliyun"}
GIT_MIRROR=${GIT_MIRROR:-"gitee"}

#valid conda channel
CONDA_MIRROR=${CONDA_MIRROR:-"tuna"}
support_conda_mirrors=("tuna" "anaconda")
if [[ " ${support_conda_mirrors[@]} " =~ " ${CONDA_MIRROR} " ]]; then
    echo "current conda channel is ${CONDA_MIRROR}"
else
    echo "not support conda channel: ${CONDA_MIRROR}, please set CONDA_MIRROR to one of ${support_conda_mirrors[@]}"
    exit 1
fi

GIT_BYZER_LLM="https://gitee.com/allwefantasy/byzer-llm.git"
GIT_VLLM="https://gitee.com/allwefantasy/ori-vllm.git"
GIT_AVIARY="https://gitee.com/allwefantasy/aviary.git"
GIT_OPTIMUM="https://gitee.com/allwefantasy/optimum.git"
GIT_AVIARY_DEEPSPEED="https://gitee.com/allwefantasy/DeepSpeed.git@aviary"
GIT_TGI="https://gitee.com/mirrors/text-generation-inference.git"
GIT_PEFT="https://gitee.com/allwefantasy/peft.git"

if [[ "${GIT_MIRROR}" == "github" ]]; then
    GIT_BYZER_LLM="https://github.com/allwefantasy/byzer-llm.git"
    GIT_VLLM="https://github.com/vllm-project/vllm.git"
    GIT_AVIARY="https://github.com/ray-project/aviary.git"
    GIT_OPTIMUM="https://github.com/huggingface/optimum.git"
    GIT_AVIARY_DEEPSPEED="https://github.com/Yard1/DeepSpeed.git@aviary"
    GIT_TGI="https://github.com/huggingface/text-generation-inference.git"
    GIT_PEFT="https://github.com/huggingface/peft.git"
fi

RAY_DASHBOARD_HOST=${RAY_DASHBOARD_HOST:-"0.0.0.0"}

# --- define pypi download mirror ---
setup_pypi_mirror() {
  echo "Setup pip mirror"

  if [[ ! -d "$HOME/.pip" ]]; then
      mkdir -p ~/.pip
  fi
  
  if [ ${PYPI_MIRROR} == "aliyun" ]; then
    cat <<EOF > ~/.pip/pip.conf
[global]
 trusted-host = mirrors.aliyun.com
 index-url = https://mirrors.aliyun.com/pypi/simple
EOF
  elif [ ${PYPI_MIRROR} == "tsinghua" ]; then
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  fi
}

cat <<EOF
This script will help you install Byzer-LLM enviroment on your machine (CentOS 8 or Ubuntu 20.04/22.04)

You should execute this script twice, first time as root user, second time as byzerllm user.
The first time, this script create a user byzerllm.
The second time, this script install the byzer-llm enviroment for byzerllm user.

1. Byzer-lang Version: ${BYZER_VERSION}
2. Byzer-notebook Version: ${BYZER_NOTEBOOK_VERSION}
EOF

# check USER_PASSWORD is set or not
if [[ -z "${USER_PASSWORD}" && "${USER}" != "byzerllm" ]]; then
    echo "We will try to create a user byzerllm  in this Machine. You should specify the USER_PASSWORD of byzerllm first"
    echo ""
    echo "Please input the USER_PASSWORD of byzerllm: "
    echo ""
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

if command -v wget &> /dev/null; then
    echo "wget is installed"    
else
    echo "wget is not installed, now install wget"    
    if [ "$OS" = "ubuntu" ]; then
        apt install -y wget
    elif [ "$OS" = "centos" ]; then
        yum install -y wget
    fi
fi

if command -v curl &> /dev/null; then
    echo "curl is installed"    
else
    echo "curl is not installed, now install curl"    
    if [ "$OS" = "ubuntu" ]; then
        apt install -y curl
    elif [ "$OS" = "centos" ]; then
        yum install -y curl
    fi
fi

if command -v ifconfig &> /dev/null; then
    echo "ifconfig is installed"    
else
    echo "ifconfig is not installed, now install ifconfig"    
    if [ "$OS" = "ubuntu" ]; then
        sudo apt install -y net-tools
    elif [ "$OS" = "centos" ]; then
        sudo yum install -y net-tools
    fi
fi


echo "Setup basic user byzerllm "

if id -u byzerllm >/dev/null 2>&1; then
    echo "User exists"
else
    echo "User byzerllm does not exist"
    groupadd ai
    useradd -m byzerllm -g ai -s /bin/bash
    echo "byzerllm:${USER_PASSWORD}" | sudo chpasswd
    echo "Grant user sudo permission"
    echo "byzerllm ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    cat <<EOF
You have successfully setup a new user byzerllm in this machine.
Please try to execute this script again in byzerllm user.
EOF
    exit 0
fi
    
if [[ "${USER}" != "byzerllm" ]];then
    echo "Please try to execute this script again in byzerllm user."
    exit 0
fi

echo "Setup basic environment for byzerllm user"
echo "go to home directory"
cd ~

echo "Install Conda environment"

CONDA_INSTALL_PATH=$HOME/miniconda3
CONDA_PREFIX=$CONDA_INSTALL_PATH
CONDA_COMMAND=$CONDA_INSTALL_PATH/bin/conda

echo "Download the latest version of Miniconda"

if [[ -d "$HOME/miniconda3" ]]; then
    echo "Miniconda is already installed"
else
    if [[ $CONDA_MIRROR == "anaconda" ]]; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    elif [[ $CONDA_MIRROR == "tuna" ]]; then
        wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    fi
    chmod +x ~/miniconda.sh
    ./miniconda.sh -b -p $HOME/miniconda3    
fi

# echo "export PATH=\"$CONDA_INSTALL_PATH/bin:\$PATH\"" >> ~/.bashrc
echo "Initialize conda and activate the base environment"
$CONDA_INSTALL_PATH/bin/conda init bash
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

echo "Create conda environments: byzerllm-dev"

if ${CONDA_COMMAND} env list | grep -q "^byzerllm-dev "; then
    echo "Conda environment byzerllm-dev exists"
else
    ${CONDA_COMMAND} create -y --name byzerllm-dev python=3.10.11
fi

source $CONDA_PREFIX/bin/activate byzerllm-dev

setup_pypi_mirror

echo "Now install the NVIDIA toolkit with conda"

# for now pytorch use cuda 11.8.0 by default.
# We should update this version when pytorch update the default version
${CONDA_COMMAND} install -y cuda -c nvidia/label/cuda-11.8.0

if [[ $CUDA_DNN_SUPPORT == "true" ]]; then
    ${CONDA_COMMAND} install -y libcublas -c nvidia/label/cuda-11.8.0
    ${CONDA_COMMAND} install -y cudnn -c nvidia/label/cuda-11.8.0 
fi



if command -v nvcc &> /dev/null; then
    echo "NVIDIA toolkit is installed from conda"
else
    echo "Fail to install NVIDIA toolkit from conda"
    exit 1
fi

echo "Create some basic folders: models projects byzerllm_stroage softwares data"

for dir in models projects byzerllm_stroage softwares data; do
    if [[ -d "$HOME/$dir" ]]; then
        echo "$dir is already created"
    else
        mkdir $HOME/$dir
    fi
done


echo "Install some basic python packages"
if [[ -d "byzer-llm" ]]; then
    echo "byzer-llm project is already exists"
else    
    git clone ${GIT_BYZER_LLM}
fi


pip install "git+${GIT_PEFT}"
pip install -r byzer-llm/demo-requirements.txt

# echo install byzer-llm package
cd byzer-llm
pip install .
cd ..

# in some cuda version, the 9.0 is not supported, if that case, try to remove 9.0 from TORCH_CUDA_ARCH_LIST_VALUE
TORCH_CUDA_ARCH_LIST_VALUE="8.0 8.6 9.0"
if [[ "${TGI_SUPPORT}" == "true" ]]; then
    echo "Setup TGI support in Byzer-LLM"
    export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
    export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
    wget "https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init" && chmod +x rustup-init && ./rustup-init -y && rm rustup-init && source "$HOME/.cargo/env"
    # source "$HOME/.cargo/env" && PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && rm -f $PROTOC_ZIP    
    source "$HOME/.cargo/env" && PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && curl -OL https://gitee.com/allwefantasy/byzer-llm/releases/download/dependency-protoc/$PROTOC_ZIP && sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && rm -f $PROTOC_ZIP    
    source "$HOME/.cargo/env" && pip install tensorboard ninja text-generation
    source "$HOME/.cargo/env" && export FORCE_CUDA=1 && TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VALUE} && git clone ${GIT_TGI} && cd text-generation-inference && git checkout 5e6ddfd6a4fecc394255d7109f87c420c98b4e15 && BUILD_EXTENSIONS=True make install
    source "$HOME/.cargo/env" && export FORCE_CUDA=1 && TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VALUE} && cd text-generation-inference/server && BUILD_EXTENSIONS=True make install-flash-attention
    source "$HOME/.cargo/env" && export FORCE_CUDA=1 && TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VALUE} && cd text-generation-inference/server && BUILD_EXTENSIONS=True make install-flash-attention-v2
    source "$HOME/.cargo/env" && export FORCE_CUDA=1 && TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VALUE} && cd text-generation-inference/server && make install-vllm   
    # if pip show custom-kernels >/dev/null 2>&1; then
    #     echo "Package custom-kernels is already installed"
    # else
    #     echo "Package custom-kernels is not installed"
    #     echo "Try to install custom-kernels"
    #     git clone https://gitee.com/mirrors/text-generation-inference
    #     cd  text-generation-inference/server/custom_kernels
    #     pip install .
    # fi 

    # cd ~/byzer-llm
    
    # if pip show flash-attn >/dev/null 2>&1; then
    #     echo "Package flash-attn is already installed"
    # else
    #     echo "Package flash-attn is not installed"
    #     echo "Install tgi flash attention dependency, it may take a while"
    #     make install-flash-attention
    # fi 

    # if pip show vllm >/dev/null 2>&1; then
    #     echo "Package vllm is already installed"
    # else
    #     echo "Package vllm is not installed"
    #     echo "Install TGI vllm dependency it may take a while "
    #     make install-vllm
    # fi 
fi  

if [[ "${VLLM_SUPPORT}" == "true" ]]; then
    echo "Setup VLLM support in Byzer-LLM"
    pip install --no-deps "git+${GIT_VLLM}"    
fi

if [[ "${AVIARY_SUPPORT}" == "true" ]]; then
    pip uninstall -y ray && pip install -U https://gitee.com/allwefantasy/byzer-llm/releases/download/dependency-ray-3.0.0/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl
    export FORCE_CUDA=1 NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_OPS=1 DS_BUILD_AIO=0 DS_BUILD_SPARSE_ATTN=0 ${TORCH_CUDA_ARCH_LIST_VALUE} && pip install \
  "awscrt" \
  "Jinja2" \
  "numexpr>=2.7.3" \
  "git+${GIT_AVIARY_DEEPSPEED}" \
  "numpy<1.24" \
  "ninja"
    pip install --no-deps "git+${GIT_OPTIMUM}"
    pip install --no-deps "git+${GIT_AVIARY}"
fi

## When use deepspeed inference, it will throws RuntimeError: Error building extension 'transformer_inference'. 
## This is because the /home/byzerllm/miniconda3/envs/byzerllm-dev has no lib64, we should make a soft link to lib to fix this issue.
## You can use the following command to reproduce this issue
# python -c """
# import deepspeed
# import transformers
# import os
# model = transformers.AutoModelForCausalLM.from_pretrained('/home/byzerllm/models/llama-7b-cn')
# world_size = int(os.getenv('WORLD_SIZE', '1'))
# model = deepspeed.init_inference(
#             model,
#             mp_size=world_size,
#             replace_with_kernel_inject=True,
#             replace_method='auto',
#         )
# """

if [[ -d "$CONDA_INSTALL_PATH/envs/byzerllm-dev/lib64" ]]; then
    echo "lib64 is already exists"
else
    echo "$CONDA_INSTALL_PATH/envs/byzerllm-dev/lib64 in is not exists,this will cause RuntimeError: Error building extension 'transformer_inference' when use deepspeed inference"
    echo "Try to create a soft link to lib64"
    ln -s $CONDA_INSTALL_PATH/envs/byzerllm-dev/lib $CONDA_INSTALL_PATH/envs/byzerllm-dev/lib64
fi


cd $HOME/softwares

if [[ $ROLE == "master" ]];then
    echo "Since we are in master mode, we should install Byzer Lang and Byzer Notebook"
    
    if [[ ! -f "byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}.tar.gz" ]]; then
       wget "https://download.byzer.org/byzer/byzer-lang/${BYZER_VERSION}/byzer-lang-all-in-one-linux-amd64-3.3.0-2.3.8.tar.gz" -O byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}.tar.gz    
    fi

    tar -zxvf byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}.tar.gz    
    BYZER_LANG_HOME=$HOME/softwares/byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}
    
    if [[ ! -f "byzer-notebook-${BYZER_NOTEBOOK_VERSION}.tar.gz" ]]; then
        wget "https://download.byzer.org/byzer/byzer-notebook/${BYZER_NOTEBOOK_VERSION}/byzer-notebook-${BYZER_NOTEBOOK_VERSION}.tar.gz" -O byzer-notebook-${BYZER_NOTEBOOK_VERSION}.tar.gz
    fi    
    tar -zxvf byzer-notebook-${BYZER_NOTEBOOK_VERSION}.tar.gz
    BYZER_NOTEBOOK_HOME=$HOME/softwares/byzer-notebook

    echo "Setup JDK"
    # On some Linux distributions, such as Ubuntu 22.04, the contents of the ~/.bashrc file are not executed \
    # in non-interactive mode, so we need to explicitly set the variables JAVA_HOME and PATH in the script
    export JAVA_HOME=${BYZER_LANG_HOME}/jdk8
    export PATH=${JAVA_HOME}/bin:$PATH
    
    cat <<EOF >> ~/.bashrc
export JAVA_HOME=${BYZER_LANG_HOME}/jdk8
export PATH=\${JAVA_HOME}/bin:\$PATH
EOF

    source ~/.bashrc

    echo "Setup MySQL"
    if command -v docker &> /dev/null; then
        echo "docker is installed"    
    else
        echo "docker is not installed, now install docker"    
        if [ "$OS" = "ubuntu" ]; then
            sudo apt install -y docker.io || STATUS=$?
            if [ $STATUS -eq 0 ]; then
                echo "install docker.io succeeded"
            else
                echo "change docker source to docker.com"
                sudo apt-get update
                sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
                curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
                sudo add-apt-repository -y "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
                sudo apt-get update
                sudo apt-get install -y docker-ce
                sudo systemctl start docker 
                sudo systemctl enable docker 
            fi          
        elif [ "$OS" = "centos" ]; then
            sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
            sudo dnf install -y docker-ce docker-ce-cli containerd.io
            sudo systemctl start docker
            sudo systemctl enable docker
        fi
    fi



    echo "Start MySQL"

    if sudo docker ps -a | grep -q "metadb"; then
        echo "docker is already running"
    else 
        echo "docker is not running, now start docker"
        MAX_RETRIES=3
        RETRY_DELAY=5
        for i in $(seq 1 $MAX_RETRIES); do
            sudo docker run --name metadb -e MYSQL_ROOT_PASSWORD=${DEFUALT_MYSQL_PASSWORD} -p 3306:3306 -d mysql:5.7 && break
            echo "Failed to start container. Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
        done
    fi
    

    
    echo "Setup Ray"

    cat <<EOF > ~/softwares/ray.start.master.sh
ray stop && ray start --head --dashboard-host ${RAY_DASHBOARD_HOST} '--resources={"master": 1, "passwordless_ssh_node": 1000}'
EOF
    
    chmod u+x ~/softwares/ray.start.master.sh

    echo "Start Ray"
    ~/softwares/ray.start.master.sh
        
    # echo "Modify the byzer config file"    
    cat <<EOF > ${BYZER_LANG_HOME}/conf/byzer.properties.override
byzer.server.mode=all-in-one
byzer.server.dryrun=false

byzer.server.runtime.driver-memory=24g

streaming.name=Byzer-lang-engine
streaming.rest=true
streaming.thrift=false
streaming.platform=spark
streaming.spark.service=true
streaming.job.cancel=true
streaming.datalake.path=./data/
streaming.driver.port=9003
streaming.enableHiveSupport=false
streaming.plugin.clzznames=tech.mlsql.plugins.ds.MLSQLExcelApp,tech.mlsql.plugins.assert.app.MLSQLAssert,tech.mlsql.plugins.shell.app.MLSQLShell,tech.mlsql.plugins.mllib.app.MLSQLMllib,tech.mlsql.plugins.llm.LLMApp,tech.mlsql.plugins.execsql.ExecSQLApp

spark.mlsql.log.driver.enablePrint=true
spark.mlsql.path.schemas=oss,s3a,s3,abfs,file
spark.mlsql.session.expireTime=10d
spark.local.dir=/home/byzerllm/byzerllm_stroage    

EOF

wget https://download.byzer.org/byzer-extensions/nightly-build/byzer-llm-3.3_2.12-0.1.0-SNAPSHOT.jar -O $BYZER_LANG_HOME/plugin/byzer-llm-3.3_2.12-0.1.0-SNAPSHOT.jar

# echo "Modify the byzer notebook config file"

cat <<EOF > ${BYZER_NOTEBOOK_HOME}/conf/notebook.properties
notebook.logo=${NOTEBOOK_LOGO}
notebook.port=9002
notebook.session.timeout=12h
notebook.security.key=6173646661736466e4bda0e8bf983161
notebook.services.communication.token=6173646661736466e4bda0e8bf983161

notebook.database.type=mysql
notebook.database.ip=localhost
notebook.database.port=3306
notebook.database.name=notebook
notebook.database.username=root
notebook.database.password=${DEFUALT_MYSQL_PASSWORD}
notebook.user.home=/home/byzerllm/data/notebook
notebook.url=http://localhost:9002
notebook.mlsql.engine-url=http://127.0.0.1:9003
notebook.mlsql.engine-backup-url=http://127.0.0.1:9003
notebook.mlsql.auth-client=streaming.dsl.auth.client.DefaultConsoleClient

notebook.job.history.max-size=2000000
notebook.job.history.max-time=30
notebook.env.is-trial=false

EOF

echo "Start Byzer lang"
cd $BYZER_LANG_HOME
./bin/byzer.sh restart

sleep 10
cd $BYZER_NOTEBOOK_HOME
echo "Start Byzer notebook"

./bin/notebook.sh restart

    cat <<EOF
1. The byzer-lang is installed at ${BYZER_LANG_HOME}
   1.1 Use ./conf/byzer.properties.override to config byzer-lang
   1.2 Use ./bin/byzer.sh start to start byzer-lang

2. The byzer-notebook is installed at ${BYZER_NOTEBOOK_HOME}
   3.1 Use ./conf/notebook.properties to config byzer-notebook
   3.2 Use ./bin/notebook.sh start to start byzer-notebook

3. ray start script is installed at $HOME/softwares/ray.start.master.sh
   4.1 You can use bash ray.start.master.sh to start ray cluster
   4.2 You can use bash ray.start.worker.sh to start ray worker

4. Please according to the https://docs.byzer.org/#/byzer-lang/zh-cn/byzer-llm/deploy to setup the byzer-lang and byzer-notebook
EOF

fi


if [[ $ROLE == "worker" ]];then

    cat <<EOF > ~/softwares/ray.start.worker.sh
echo -n "The master ip is: "
read masterIP
echo -n "The name of this worker is: "
read workerName
ray stop && ray start --address="\${masterIP}:6379"  "--resources={\"\${workerName}\": 1}"
EOF

chmod u+x ~/softwares/ray.start.worker.sh

    cat <<EOF
ray start script is installed at $HOME/softwares/ray.start.worker.sh
    You can use \`bash ray.start.master.sh\` to start ray cluster
    You can use \`bash ray.start.worker.sh\` to start ray worker
EOF

fi