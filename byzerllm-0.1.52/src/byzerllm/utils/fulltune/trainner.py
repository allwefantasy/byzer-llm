import json
import os

import deepspeed
import deepspeed.comm as dist
import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from pyjava import RayContext

from transformers import Trainer, TrainingArguments,default_data_collator

from .base_model.configuration_baichuan import BaiChuanConfig
from .base_model.modeling_baichuan import BaiChuanForCausalLM
from . import TrainParameters
from ray.train.huggingface import TransformersTrainer
from ray.air.config import ScalingConfig
from ray.air import session

deepspeed_confg = json.loads('''
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
}''')
                             
class DataEngine():
    def __init__(self, data_dir, tokenizer_path, micro_batch_size, max_length):
        self.MIN_TEXT_LEN = 20
        self.EOS_TOKEN_ID = 2
        self.data_dir = data_dir
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        self.micro_batch_size = micro_batch_size
        self.max_length = max_length
        self.data = []
        self.global_input_paths = [self.data_dir + "/" + x
                                   for x in os.listdir(self.data_dir)]
        self.local_input_paths = [x for i, x in enumerate(self.global_input_paths)]

    def load_data(self):
        for file_path in self.local_input_paths:
            data = []
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line_id, line in enumerate(f):
                    line_json = json.loads(line.strip())
                    text = line_json("instruction") + line_json("input") + line_json("output")
                    cc = self.sp.EncodeAsIds(text) + [self.EOS_TOKEN_ID]
                    if len(cc) < self.MIN_TEXT_LEN:
                        cc = []
                    data.extend(cc)
                    if len(data) >= self.micro_batch_size * (self.max_length + 1):
                        index = self.micro_batch_size * (self.max_length + 1)
                        self.data.append(data[:index])
                        data = []
        return

    def get_data(self):
        data = self.data.pop(0)
        seq = np.asarray(data).reshape(self.micro_batch_size, self.max_length + 1)
        data = torch.LongTensor(seq)
        # data = data.cuda(non_blocking=True)
        return data                             
                             
class FulltuneDataset(Dataset):

    def __init__(self,data_engine:DataEngine) -> None:
        self.data_engine = data_engine


    def __len__(self):
        return len(self.data_engine.data)

    def __getitem__(self, index):                
        input_ids = self.data_engine.get_data()
        inputs = {
            'input_ids': input_ids            
        }
        return inputs                       


def trainer_init_per_worker(train_dataset, eval_dataset=None, **config):
    # Use the actual number of CPUs assigned by Ray
    os.environ["OMP_NUM_THREADS"] = str(
        session.get_trial_resources().bundles[-1].get("CPU", 1)
    )
    # Enable tf32 for better performance
    # torch.backends.cuda.matmul.allow_tf32 = True

    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 2)
    warmup_steps = config.get("warmup_steps", 0)
    learning_rate = config.get("learning_rate", 0.00002)
    weight_decay = config.get("weight_decay", 0.01)

    print("Preparing training arguments")
    training_args = TrainingArguments(
        "output",
        per_device_train_batch_size=batch_size,
        logging_steps=1,
        save_strategy="no",
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        label_names=["input_ids", "attention_mask"],
        num_train_epochs=epochs,
        push_to_hub=False,
        disable_tqdm=True,  # declutter the output a little
        fp16=True,
        gradient_checkpointing=True,
        deepspeed=deepspeed_confg,
    )
    
    data_refs = config["data_refs"]
    train_args = config["train_args"] 
    index = 0 # dist.get_rank()
        
    if not os.path.exists(os.path.join(train_args.data_dir,str(index))):
        os.makedirs(os.path.join(train_args.data_dir,str(index)))
    
    data_file_path = os.path.join(train_args.data_dir,str(index),"data.json")
    with open(data_file_path,"w",encoding="utf-8") as f:        
        for item in RayContext.collect_from([data_refs[index]]):
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
                                
    micro_batch_size =  deepspeed_confg["train_micro_batch_size_per_gpu"]
    data_engine = DataEngine(train_args.data_dir, train_args.tokenizer_path, micro_batch_size, train_args.max_length)
    data_engine.load_data()   
    fulltune_dataset = FulltuneDataset(data_engine) 

    os.remove(data_file_path)

    model = BaiChuanForCausalLM(BaiChuanConfig())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=fulltune_dataset        
    )
    return trainer



def distribute_train(args:TrainParameters,data_refs):

    assert  args.num_workers == len(data_refs), f"num_workers({args.num_workers}) must equal to data_refs({len(data_refs)})"    

    distribute_trainer = TransformersTrainer(
                                trainer_init_per_worker=trainer_init_per_worker,
                                trainer_init_config={
                                    "batch_size": 16,  # per device
                                    "epochs": 1,
                                    "data_refs":data_refs,
                                    "train_args":args
                                },
                                scaling_config=ScalingConfig(
                                    num_workers=args.num_workers,
                                    use_gpu=args.use_gpu,
                                    resources_per_worker={"GPU": args.gpus_per_worker, "CPU": args.cpus_per_worker},
                                )                                                                                               
    ) 
    r = distribute_trainer.fit()
    return r.path()

