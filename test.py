import ray
import numpy as np
from pyjava.api.mlsql import RayContext
from byzerllm.chatglm6b.finetune import restore_model,load_model,init_model,predict
from pyjava.udf import UDFMaster,UDFWorker,UDFBuilder,UDFBuildInFunc

from typing import Any, NoReturn, Callable, Dict, List
import time
from ray.util.client.common import ClientActorHandle, ClientObjectRef
import uuid
import os
from byzerllm.chatglm6b.finetune import finetune_or_infer,restore_model,load_model
from byzerllm.chatglm6b.arguments import ModelArguments,DataTrainingArguments
from transformers import Seq2SeqTrainingArguments

MODEL_DIR="/my8t/tmp/adgen-chatglm-6b-pt-8-1e-2/checkpoint-100"
OUTPUT_DIR="/my8t/tmp/checkpoint"
DATA_FILE="/tmp/traindata.json"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_args = ModelArguments(    
    model_name_or_path=MODEL_DIR,    
    pre_seq_len=8    
)
# pre_seq_length=8,
training_args = Seq2SeqTrainingArguments(
    do_train=False,
    do_eval=False,
    do_predict=True,
    overwrite_output_dir=True,
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,        
    predict_with_generate=True
)

data_args = DataTrainingArguments(
    max_source_length=64,
    max_target_length=64,
    prompt_column="instruction",
    response_column="output",
    train_file=DATA_FILE,
    validation_file=DATA_FILE,    
    overwrite_cache=True
)

data = [
    {
        "instruction":"类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞",
        "output":"简约而不简单的牛仔外套，白色的衣身十分百搭。衣身多处有做旧破洞设计，打破单调乏味，增加一丝造型看点。衣身后背处有趣味刺绣装饰，丰富层次感，彰显别样时尚。"
    }
]

(trainer,tokenizer)=init_model(model_args,data_args,training_args)
results = predict(data, data_args,training_args, trainer, tokenizer)
import json
print(json.dumps(results,ensure_ascii=False,indent=4))
