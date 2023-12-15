import subprocess
import os
import ray
import json
import paramiko
import dataclasses
import uuid
import hashlib
from pyjava import RayContext
from typing import List
import getpass
from byzerllm import transfer_from_ob
from pyjava.udf.store import transfer_to_ob
from pyjava.storage import streaming_tar as STar

@dataclasses.dataclass
class TrainParameters():
    name:str
    data_dir:str
    config_dir:str    
    tokenizer_path:str
    model_dir:str
    max_length:int=4096
    steps_per_epoch:int=4096
    checkpoint_saving_path:str="checkpoints"     
    num_workers:int=1
    use_gpu:bool=True
    cpus_per_worker:int=1
    gpus_per_worker:int=1
    user:str="byzerllm"
    password:str="byzerllm"
    ssh_port:int=22
    passwordless_ssh_node:int=0
    auto_setup_passwordless_ssh:bool=False 
    private_key_name:str="byzerllm_id_rsa"
    public_key_name:str="byzerllm_id_rsa.pub" 
          

base_deepspeed_cnofig = json.loads('''
{
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1,
  "prescale_gradients": false,
  "zero_allow_untested_optimizer": true,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-8,
      "eps": 1.0e-8,
      "betas": [
        0.9,
        0.95
      ],
      "weight_decay": 0.1
    }
  },
  "tensorboard": {
    "enabled": true,
    "output_path": "logs/",
    "job_name": "baichuan-7b-pt"
  },
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": false,
    "allgather_bucket_size": 3e8,
    "reduce_bucket_size": 3e8,
    "overlap_comm": true,
    "reduce_scatter": true
  },
  "steps_per_print": 16,
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": true,
  "bf16": {
    "enabled": true
  }
}

''')

def exec_command(command):
    return start_command(command)

def start_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Redirect stdout and stderr to the main Python process
    while True:
        output = process.stdout.readline().decode().strip()
        error = process.stderr.readline().decode().strip()
        if output:
            print(output,flush=True)
        if error:
            print(error,flush=True)
        if output == '' and error == '' and process.poll() is not None:
            break
    
    process.stdout.close()
    process.stderr.close()
    return process.poll()

def get_cmd(config_dir:str, client_args:str="", has_hostfile=False):
    hostfile_chunk = f" --hostfile={os.path.join(config_dir,'hostfile')} " if has_hostfile else ""
    include = ""
    if not has_hostfile:
        gpus = ray.get_gpu_ids()
        include = f" --include=localhost:{','.join(gpus)}"
    return f"deepspeed {include} {hostfile_chunk} --module byzerllm.utils.fulltune.launch {client_args}  --deepspeed --deepspeed_config {os.path.join(config_dir,'deepspeed.json')}"


def check_passwordless_ssh(ipOrHost):
    import paramiko
    import paramiko.util    

    # Load the SSH config file
    ssh_config = paramiko.SSHConfig()
    with open(os.path.expanduser('~/.ssh/config')) as f:
        ssh_config.parse(f)
    
    # status = exec_command(f"ssh {ipOrHost}")
    # return status == 0

    user_config = ssh_config.lookup(ipOrHost)
    return "user" in user_config
    

def try_connect_and_get_fingerprint(target_ip):
    # try ssh -o StrictHostKeyChecking=accept-new 6baf1a94a956f208af8ca0e3dc887d64
    s = subprocess.check_output(f"ssh-keyscan -t rsa {target_ip}", shell=True, universal_newlines=True)    
    return s
    

def setup_passwordless_ssh(worker,host,ip,port, username, password):
    _username = username if username else getpass.getuser()
    home_path = os.path.expanduser("~") 
    
    if not os.path.exists(os.path.join(home_path,".ssh","byzerllm_id_rsa")): 
        key = paramiko.RSAKey.generate(bits=2048)    

        if not os.path.exists(os.path.join(home_path,".ssh")):
            os.makedirs(os.path.join(home_path,".ssh"))

        key.write_private_key_file(os.path.join(home_path,".ssh","byzerllm_id_rsa"))

        public_key = key.get_base64()
        with open(os.path.join(home_path,".ssh","byzerllm_id_rsa.pub"), "w") as f:
            f.write(f"ssh-rsa {public_key}")
    
    with open(os.path.join(home_path,".ssh","byzerllm_id_rsa.pub"), "r") as f:
        public_key = f.read() 
        
    ssh_config = f'''
Host {host}
HostName {ip}
User {_username}
Port {port}
IdentityFile ~/.ssh/byzerllm_id_rsa
    '''

    exec_command(f'''
cat <<EOF >> ~/.ssh/config
{ssh_config}
EOF''')
    ray.get(worker.setup_pubkey.remote(public_key))
    
    fingerprint = try_connect_and_get_fingerprint(ip)
    exec_command(f'''
cat <<EOF >> ~/.ssh/known_hosts
{fingerprint}
EOF''')
    
    s = check_passwordless_ssh(host)    
    if not s:
        raise Exception(f"setup passwordless ssh failed in {ip}")

    

def encode_str(input:str):
    encoded_string = hashlib.md5(input.encode("utf-8")).hexdigest()    
    return encoded_string

@ray.remote
class TrainMaster():
    def __init__(self,args:TrainParameters,data_refs,model_refs,standalone) -> None:
        self.standalone = standalone
        self.args = args
        self.data_refs = data_refs
        self.data_ref = data_refs[0]        
        self.workers = []
        self.ips = []
        self.ipToHost = {}
        self.ipToWorkers = {}
        self.model_refs = model_refs
        if not standalone:
            self.workers = [TrainWorker.options(num_cpus=args.cpus_per_worker,
                                    num_gpus=args.gpus_per_worker).remote(args,data_refs[i],model_refs) for i in range(self.args.num_workers)]
            self.ips = ray.get([worker.get_ip.remote() for worker in self.workers]) 
            self.ipToHost = {ip:encode_str(ip) for ip in self.ips}
            for ip,worker in zip(self.ips,self.workers):
                if ip not in self.ipToWorkers:                    
                    self.ipToWorkers[ip] = []
                self.ipToWorkers[ip].append(worker)
        
        self.id = str(uuid.uuid4())

    def get_model(self):
        if self.standalone:         
            new_model_refs = []
            transfer_to_ob(self.args.name,self.args.checkpoint_saving_path,new_model_refs)
            return new_model_refs    
        else:
            return ray.get(self.worker[0].get_model.remote())

    def shutdown(self):
        if len(self.workers) > 0:
            shutdown_jobs = [ray.kill(worker) for worker in self.workers]
            ray.get(shutdown_jobs)
        ray.actor.exit_actor()

    def fit(self):
        
        if not os.path.exists(self.args.config_dir):
            os.makedirs(self.args.config_dir)
            
        with open(os.path.join(self.args.config_dir,"hostfile"),"w") as f:
            temp_ips = set()           
            for ip in self.ips:
                # Todo: we should set gpus ids from ray.get_gpu_ids() in every worker in feature
                # so the gpus used in deepspeed match the gpus used in ray
                workers = self.ipToWorkers[ip]
                if ip not in temp_ips:
                    temp_ips.add(ip)
                    f.write(f"{self.ipToHost[ip]} slots={self.args.gpus_per_worker * len(workers)}\n")

        with open(os.path.join(self.args.config_dir,"deepspeed.json"),"w") as f:
            f.write(json.dumps(base_deepspeed_cnofig))

        real_data_dir = self.args.data_dir
        
        # if standalone, we should save data to in master 
        if self.standalone:            
            if not os.path.exists(real_data_dir):
                os.makedirs(real_data_dir)
        
            data_file_path = os.path.join(real_data_dir,f"data-{self.id}.jsonl")
            with open(data_file_path,"w",encoding="utf-8") as f:        
                for item in RayContext.collect_from([self.data_ref]):
                    f.write(json.dumps(item,ensure_ascii=False)+"\n")
        
        # if not standalone, we should save data to every worker for deepspeed worker
        if len(self.workers) > 0:
            prepare_data_jobs = [worker.fit.remote() for worker in self.workers]
            ray.get(prepare_data_jobs)            

        client_args = {
            "data_dir":real_data_dir,
            "tokenizer_path":self.args.tokenizer_path,
            "checkpoint_saving_path":self.args.checkpoint_saving_path,
        }
        temp_clien_args = []
        for k,v in client_args.items():
            temp_clien_args.append(f"--{k} {v}")

        command = get_cmd(self.args.config_dir," ".join(temp_clien_args),has_hostfile=len(self.ips)>0)
        print(f"[{self.args.name}] deepspeed command: {command}")
        start_command(command)

    def setup_worker(self):    
        if not self.args.auto_setup_passwordless_ssh:
            return
        
        my_ip = self.get_ip()    
        for ip in self.ips:  
            hostname = self.ipToHost[ip] 
            workers = self.ipToWorkers[ip]         
            if not check_passwordless_ssh(hostname):                
                    print(f"[{self.args.name}] try to automatically setup ssh passwordless in {ip} from {my_ip}",flush=True)
                    setup_passwordless_ssh(workers[0],hostname,ip,self.args.ssh_port,self.args.user,self.args.password)                                                               
               

    def get_ip(self):
        id = ray.get_runtime_context().get_node_id()
        ip = [node for node in ray.nodes() if node['NodeID']==id][0]['NodeManagerAddress']
        return ip

@ray.remote
class TrainWorker():
    def __init__(self,args:TrainParameters,data_ref,model_refs) -> None:
        self.data_ref = data_ref
        self.args = args        
        self.id = str(uuid.uuid4()) 
        self.model_refs = model_refs


    def get_model(self):
        new_model_refs = []
        transfer_to_ob(self.args.name,self.args.checkpoint_saving_path,new_model_refs)
        return new_model_refs

    def fit(self):
        real_data_dir = self.args.data_dir

        if not os.path.exists(real_data_dir):
            os.makedirs(real_data_dir)
    
        data_file_path = os.path.join(real_data_dir,f"data-{self.id}.jsonl")
        with open(data_file_path,"w",encoding="utf-8") as f:        
            for item in RayContext.collect_from([self.data_ref]):
                f.write(json.dumps(item,ensure_ascii=False)+"\n")
        
        if len(self.model_refs) > 0:
            transfer_from_ob(self.args.name,self.args.model_refs,self.args.model_dir)
                
    def setup_pubkey(self,pubkey):
        home_path = os.path.expanduser("~") 
        if not os.path.exists(os.path.join(home_path,".ssh")):
            os.makedirs(os.path.join(home_path,".ssh"))
        status = exec_command('echo "{}" >> ~/.ssh/authorized_keys'.format(pubkey))
        exec_command("chmod 700 ~/.ssh")
        exec_command("chmod 600 ~/.ssh/authorized_keys")
        if status != 0:
            raise Exception(f"setup pubkey failed with status {status} in ${self.get_ip()}")

    def get_gpus(self):
        return ray.get_gpu_ids()
    
    def get_ip(self):
        id = ray.get_runtime_context().get_node_id()
        ip = [node for node in ray.nodes() if node['NodeID']==id][0]['NodeManagerAddress']
        return ip   

def distribute_train(args:TrainParameters,data_refs,model_refs):
    
    assert  args.num_workers == len(data_refs), f'''
    num_workers({args.num_workers}) must equal to data_refs({len(data_refs)}).
    Try to add the following code to repartition data and make sure the number of partitions equal to num_workers:
    
    ```
    run trainData as TableRepartition.`` 
    where partitionNum="{args.num_workers}" as preTrainData;
    ```

    '''  
    standalone = args.num_workers == 1

    if standalone:
        master = TrainMaster.options(num_cpus=args.cpus_per_worker,
                                        num_gpus=args.gpus_per_worker,
                                        resources={"passwordless_ssh_node":args.passwordless_ssh_node}
                                        ).remote(args,data_refs,model_refs,standalone)
    else:
        master = TrainMaster.options(num_cpus=0,
                                        num_gpus=0,
                                        resources={"passwordless_ssh_node":args.passwordless_ssh_node}
                                        ).remote(args,data_refs,model_refs,standalone) 

    new_model_refs = [] 
           
    try:
        ray.get(master.setup_worker.remote())
        ray.get(master.fit.remote())
        new_model_refs = ray.get(master.get)
    except Exception as e:
        print(f"Error: {e}")
        ray.get(master.shutdown.remote())
    
    return new_model_refs

       
    
