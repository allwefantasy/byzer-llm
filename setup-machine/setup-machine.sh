set -e # Exit immediately if any command exits with a non-zero status (i.e., an error).
set -x # Print each command before executing it, prefixed with the + character. This can be useful for debugging shell scripts.
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

if sudo -n true 2>/dev/null; then
    echo "User has sudo permission"
else
    echo "Grant user sudo permission"
    echo "byzerllm ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi
    
echo "swith to user byzerllm"
su - byzerllm

echo "Install Conda environment"

CONDA_INSTALL_PATH=$HOME/miniconda3

echo "Download the latest version of Miniconda"

if [[ -d "$HOME/miniconda3" ]]; then
    echo "Miniconda is already installed"
else
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    chmod +x ~/miniconda.sh
    ./miniconda.sh -b -p $HOME/miniconda3    
fi

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

# for now pytorch use cuda 11.7.0 by default.
# We should update this version when pytorch update the default version
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

if pip show custom-kernels >/dev/null 2>&1; then
    echo "Package custom-kernels is already installed"
else
    echo "Package custom-kernels is not installed"
    echo "Try to install custom-kernels"
    git clone https://gitee.com/mirrors/text-generation-inference
    cd  text-generation-inference/server/custom_kernels
    pip install .
fi 


cd byzer-llm

if pip show flash-attn >/dev/null 2>&1; then
    echo "Package flash-attn is already installed"
else
    echo "Package flash-attn is not installed"
    echo "Install tgi flash attention dependency, it may take a while"
    make install-flash-attention
fi 

if pip show vllm >/dev/null 2>&1; then
    echo "Package vllm is already installed"
else
    echo "Package vllm is not installed"
    echo "Install TGI vllm dependency it may take a while "
    make install-vllm
fi 


cd $HOME/softwares

if [[ $ROLE == "master" ]];then
    echo "Since we are in master mode, we should install Byzer Lang and Byzer Notebook"

    wget "https://download.byzer.org/byzer/byzer-lang/${BYZER_VERSION}/byzer-lang-all-in-one-linux-amd64-3.3.0-2.3.8.tar.gz" -O byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}.tar.gz
    tar -zxvf byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}.tar.gz
    BYZER_LANG_HOME=$HOME/softwares/byzer-lang-all-in-one-linux-amd64-3.3.0-${BYZER_VERSION}

    wget "https://download.byzer.org/byzer/byzer-notebook/${BYZER_NOTEBOOK_VERSION}/byzer-notebook-${BYZER_NOTEBOOK_VERSION}.tar.gz" -O byzer-notebook-${BYZER_NOTEBOOK_VERSION}.tar.gz
    tar -zxvf byzer-notebook-${BYZER_NOTEBOOK_VERSION}.tar.gz
    BYZER_NOTEBOOK_HOME=$HOME/softwares/byzer-notebook

    echo "Setup JDK"

    cat <<EOF >> ~/.bashrc
export JAVA_HOME=${BYZER_LANG_HOME}/jdk8
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
            sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
            sudo dnf install -y docker-ce docker-ce-cli containerd.io
        fi
    fi

    echo "Start MySQL"
    docker run --name metadb -e MYSQL_ROOT_PASSWORD=${DEFUALT_MYSQL_PASSWORD} -p 3306:3306 -d mysql:5.7

    echo "Setup Ray"

    cat <<EOF >> ~/softwares/ray.start.master.sh
ray stop && ray start --head --dashboard-host 127.0.0.1  '--resources={"master": 1, "passwordless_ssh_node": 1000}'
EOF 
    chmod u+x ~/softwares/ray.start.master.sh

    echo "Start Ray"
    ~/softwares/ray.start.master.sh
        
    # echo "Modify the byzer config file"    
    cat <<EOF > ${BYZER_LANG_HOME}/conf/byzer.properties.override
byzer.server.mode=all-in-one
byzer.server.dryrun=false

byzer.server.runtime.driver-memory=24g

streaming.name=Byzer-lang-desktop
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
notebook.env.is-trial=true

notebook.redis.host=localhost
notebook.redis.port=6379
notebook.redis.password=redis_pwd
notebook.redis.database=0

notebook.env.is-trial=true   

EOF

echo "Start Byzer lang"
cd $BYZER_LANG_HOME
./bin/byzer.sh start

sleep 10

echo "Start Byzer notebook"

./bin/notebook.sh start

    cat <<EOF
1. The byzer-lang is installed at ${BYZER_LANG_HOME}
   1.1 Use `./conf/byzer.properties.override` to config byzer-lang
   1.2 Use `./bin/byzer.sh start` to start byzer-lang

2. The byzer-notebook is installed at ${BYZER_NOTEBOOK_HOME}
   3.1 Use `./conf/notebook.properties` to config byzer-notebook
   3.2 Use `./bin/notebook.sh start` to start byzer-notebook

3. ray start script is installed at $HOME/softwares/ray.start.master.sh
   4.1 You can use `bash ray.start.master.sh` to start ray cluster
   4.2 You can use `bash ray.start.worker.sh` to start ray worker

4. Please according to the https://docs.byzer.org/#/byzer-lang/zh-cn/byzer-llm/deploy to setup the byzer-lang and byzer-notebook
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
ray start script is installed at $HOME/softwares/ray.start.worker.sh
    You can use `bash ray.start.master.sh` to start ray cluster
    You can use `bash ray.start.worker.sh` to start ray worker
EOF    

fi