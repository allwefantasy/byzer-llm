from typing import List,Dict
import json
from transformers import AutoTokenizer, BitsAndBytesConfig
from byzerllm.utils.metrics import Metric
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoModel
)
import argparse
import os
from os.path import join
import torch
import bitsandbytes as bnb
from collections import defaultdict

from .collator import SFTDataCollator
from .dataset import SFTDataset
from .argument import QLoRAArguments
from .trainer import LoRATrainer
from .loss import TargetLMLoss


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def setup_everything(lora_config:str, args:List[str]):
    parser = argparse.ArgumentParser()    
    args = parser.parse_args(args=args)
    
    # 读取训练的参数配置
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_dict(json.loads(lora_config),allow_extra_keys=True)
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


def init_components(args, training_args,extra_params):
    """
    初始化各个组件
    """
    print('Initializing components...')
    # 下面的设置至关重要，否则无法多卡训练
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    # training_args.ddp_find_unused_parameters = False if ddp else None
    device_map = "auto"
    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    # 部分tokenizer没有pad_token_id
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is None:
            print(f"tokenizer has no pad_token_id and unk_token_id, setting it to 0")
            tokenizer.pad_token_id = 0
        else:                
            print(f"tokenizer has no pad_token_id({tokenizer.pad_token_id}), setting it to unk_token_id({tokenizer.unk_token_id})")
            tokenizer.pad_token_id = tokenizer.unk_token_id
    # 如果两者相同，模型训练时不会计算eos_token_id的loss
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise Exception('pad_token_id should not be equal to eos_token_id')       
    
    # 加载模型  
    model_type = extra_params.get("model_type","casual_lm")

    if model_type == "casual_lm":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            ),
        )    
        model.tie_weights()
    else:
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            ),
        )     
    # casts all the non int8 modules to full precision (fp32) for stability
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)    
    print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
    # 找到所有需要插入adapter的全连接层
    target_modules = find_all_linear_names(model)
    # 初始化lora配置
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32

    # 查看模型种各种类型的参数的情况
    verify_model_dtype(model)

    # 初始化损失函数
    print(f'Initializing loss function (ignore_index={tokenizer.pad_token_id})...',flush=True)
    loss_func = TargetLMLoss(ignore_index=tokenizer.pad_token_id)
    
    train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length, **extra_params)
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length, **extra_params)

    # 初始化Trainer
    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_loss=loss_func
    )
    return trainer


def train(lora_config:str, args:List[str],extra_params={})->str:

    sft_name = extra_params["sft_name"]   
    # 进行一些配置和检查
    parsed_args, training_args = setup_everything(lora_config, args)
    # 加载各种组件
    trainer = init_components(parsed_args, training_args,extra_params)
    # 开始训练
    print(f"*** [{sft_name}] starting training ***")
    train_result = trainer.train()

    # 打印 token 总数
    print(f"[{sft_name}] total tokens: {trainer.train_dataset.dataset_tokens_count}",flush=True)
    token_metrics = Metric()
    token_metrics.inc(f"sft_{sft_name}_tokens_num",trainer.train_dataset.dataset_tokens_count)
    token_metrics.push()

    # 保存最好的checkpoint
    final_save_path = join(training_args.output_dir, 'final')
    trainer.save_model(final_save_path)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()  
    return final_save_path  


