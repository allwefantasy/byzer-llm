<p align="center">
  <picture>    
    <img alt="Byzer-LLM" src="https://github.com/allwefantasy/byzer-llm/blob/master/docs/source/assets/logos/logo.jpg" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap pretrain,finetune, serving for everyone
</h3>

<p align="center">
| <a href="#"><b>Documentation</b></a> | <a href="#"><b>Blog</b></a> | | <a href="#"><b>Discord</b></a> |

</p>

---

*Latest News* 🔥

- [2023/11] Release Byzer-LLM 0.1.16

---

Byzer-LLM is Ray based , a full lifecycle solution for LLM that includes pretrain, fintune, deployment and serving.

The unique features of Byzer-LLM are:

1. Full lifecyle: pretrain and finetune,deploy and serving support
2. Python/SQL API support
3. Ray based, easy to scale

---

## Versions
- 0.1.18： Support stream chat/ Support Model Template
- 0.1.17： None
- 0.1.16： Enhance the API for byzer-retrieval
- 0.1.14： add get_tables/get_databases API for byzer-retrieval
- 0.1.13: support shutdown cluster for byzer-retrieval
- 0.1.12: Support Python API (alpha)
- 0.1.5: Support python wrapper for [byzer-retrieval](https://github.com/allwefantasy/byzer-retrieval)

---

## Installation

```bash
pip install -r requirements.txt
pip install -U vllm
pip install -U byzerllm
ray start --head
```

---

## Quick Start

```python
import ray
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend

ray.init(address="auto",namespace="default",ignore_reinit_error=True)

llm = ByzerLLM()

llm.setup_gpus_per_worker(4).setup_num_workers(1)
llm.setup_infer_backend(InferBackend.transformers)

llm.deploy(model_path="/home/byzerllm/models/openbuddy-llama2-13b64k-v15",
           pretrained_model_type="custom/llama2",
           udf_name="llama2_chat",infer_params={})

llm.chat("llama2_chat",LLMRequest(instruction="hello world"))[0].output
```

The above code will deploy a llama2 model and then use the model to infer the input text. If you use transformers as the inference backend, you should specify the `pretrained_model_type` manually since the transformers backend can not auto detect the model type.

Byzer-LLM also support `deploy` SaaS model with the same way. This feature provide a unified interface for both open-source model and SaaS model. The following code will deploy a Azure OpenAI model and then use the model to infer the input text.


```python
import ray
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend
ray.init(address="auto",namespace="default",ignore_reinit_error=True)

llm = ByzerLLM()

llm.setup_gpus_per_worker(0).setup_num_workers(10)
llm.setup_infer_backend(InferBackend.transformers)

llm.deploy(pretrained_model_type="saas/azure_openai",
           udf_name="azure_openai",
           infer_params={
            "saas.api_type":"azure",
            "saas.api_key"="xxx"
            "saas.api_base"="xxx"
            "saas.api_version"="2023-07-01-preview"
            "saas.deployment_id"="xxxxxx"
           })

llm.chat("azure_openai",LLMRequest(instruction="hello world"))[0].output
```

Notice that the SaaS model does not need GPU, so we set the `setup_gpus_per_worker` to 0, and you can use `setup_num_workers`
to control max concurrency,how ever, the SaaS model has its own max concurrency limit, the `setup_num_workers` only control the max
concurrency accepted by the Byzer-LLM.

## Supported Models

The supported open-source `pretrained_model_type` are:

1. custom/llama2
2. bark	
3. whisper	
3. chatglm6b
4. custom/chatglm2
5. moss
6. custom/alpha_moss
7. dolly
8. falcon
9. llama
10. custom/starcode
11. custom/visualglm
12. custom/m3e
13. custom/baichuan
14. custom/bge
15. custom/qwen_vl_chat
16. custom/stable_diffusion
17. custom/zephyr

The supported SaaS `pretrained_model_type` are:

1. saas/chatglm	Chatglm130B
2. saas/sparkdesk	星火大模型
3. saas/baichuan	百川大模型
4. saas/zhipu	智谱大模型
5. saas/minimax	MiniMax 大模型
6. saas/qianfan	文心一言
7. saas/azure_openai	
8. saas/openai

Notice that the derived models from llama/llama2/startcode are also supported. For example, you can use `llama` to load vicuna model.

## vLLM Support

The Byzer-llm also support vLLM as the inference backend. The following code will deploy a vLLM model and then use the model to infer the input text.

```python
import ray
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend

ray.init(address="auto",namespace="default",ignore_reinit_error=True)
llm = ByzerLLM()

llm.setup_gpus_per_worker(2)
llm.setup_num_workers(1)
llm.setup_infer_backend(InferBackend.VLLM)

llm.deploy(
    model_path="/home/byzerllm/models/openbuddy-zephyr-7b-v14.1",
    pretrained_model_type="custom/auto",
    udf_name="zephyr_chat"",
    infer_params={"backend.max_num_batched_tokens":32768}
)

llm.chat("zephyr_chat",LLMRequest(instruction="hello world"))[0].output
```

There are some tiny differences between the vLLM and the transformers backend. 

1. The `pretrained_model_type` is fixed to `custom/auto` for vLLM, since the vLLM will auto detect the model type.
2. Use `setup_infer_backend` to specify `InferBackend.VLLM` as the inference backend.

If the model you deploy with the backend vLLM, then it also support `stream chat`：
the `stream_chat_oai` will return a generator, you can use the generator to get the output text.

```python
t = llm.stream_chat_oai(conversations=[{
    "role":"user",
    "content":"Hello, how are you?"
}])

for line in t:
   print(line+"\n")
```

## DeepSpeed Support

The Byzer-llm also support DeepSpeed as the inference backend. The following code will deploy a DeepSpeed model and then use the model to infer the input text.

```python
import ray
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend

ray.init(address="auto",namespace="default",ignore_reinit_error=True)
llm = ByzerLLM()

llm.setup_gpus_per_worker(4)
llm.setup_num_workers(1)
llm.setup_infer_backend(InferBackend.DeepSpeed)

llm.deploy(
    model_path="/home/byzerllm/models/openbuddy-llama-13b-v5-fp16",
    pretrained_model_type="custom/auto",
    udf_name="llama_chat"",
    infer_params={}
)

llm.chat("llama_chat",LLMRequest(instruction="hello world"))[0].output
```

The code above is totally the same as the code for vLLM, except that the `InferBackend` is `InferBackend.DeepSpeed`.

## SQL Support

In addition to the Python API, Byzer-llm also support SQL API. In order to use the SQL API, you should install Byzer-SQL language first.

Try to install the Byzer-SQL language with the following command:

```bash
git clone https://gitee.com/allwefantasy/byzer-llm
cd byzer-llm/setup-machine
sudo -i 
ROLE=master ./setup-machine.sh
```

After the installation, you can visit the Byzer Console at http://localhost:9002. 

In the Byzer Console, you can run the following SQL to deploy a llama2 model which have the same effect as the Python code above.

```sql
!byzerllm setup single;
!byzerllm setup "num_gpus=4";
!byzerllm setup "maxConcurrency=1";
!byzerllm setup "infer_backend=transformers";

run command as LLM.`` where 
action="infer"
and pretrainedModelType="custom/llama2"
and localModelDir="/home/byzerllm/models/openbuddy-llama-13b-v5-fp16"
and reconnect="false"
and udfName="llama2_chat"
and modelTable="command";

```

Then you can invoke the model with UDF `llama2_chat`:

```sql

select 
llama2_chat(llm_param(map(
              "user_role","User",
              "assistant_role","Assistant",
              "system_msg",'You are a helpful assistant. Think it over and answer the user question correctly.',
              "instruction",llm_prompt('
Please remenber my name: {0}              
',array("Zhu William"))

))) as q 
as q1;
```

Once you deploy the model with `run command as LLM`, then you can ues the model as a SQL function. This feature is very useful for data scientists who want to use LLM in their data analysis or data engineers who want to use LLM in their data pipeline.

---

### QWen

If you use QWen in ByzerLLM, you should sepcify the following parameters mannualy:

1. the role mapping 
2. the stop_token_ids
3. trim the stop tokens from the output

However, we provide a template for this, try to the following code:

```python
from byzerllm.utils.client import Templates

### Here,we setup the template for qwen
llm.setup_template("chat",Templates.qwen())

t = llm.chat_oai(conversations=[{
    "role":"user",
    "content":"你好,给我讲个100字的笑话吧?"
}])
print(t)
```

---
## SaaS Models

Since the different SaaS models have different parameters, here we provide some templates for the SaaS models to help you deploy the SaaS models.

### qianfan


```sql
!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/qianfan"
and `saas.api_key`="xxxxxxxxxxxxxxxxxx"
and `saas.secret_key`="xxxxxxxxxxxxxxxx"
and `saas.model`="ERNIE-Bot-turbo"
and `saas.retry_count`="3"
and `saas.request_timeout`="120"
and reconnect="false"
and udfName="qianfan_saas"
and modelTable="command";

```

### azure openai

```sql

!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/azure_openai"
and `saas.api_type`="azure"
and `saas.api_key`="xxx"
and `saas.api_base`="xxx"
and `saas.api_version`="2023-07-01-preview"
and `saas.deployment_id`="xxxxx"
and udfName="azure_openai"
and modelTable="command";
```

### openai

```sql

!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/azure_openai"
and `saas.api_type`="azure"
and `saas.api_key`="xxx"
and `saas.api_base`="xxx"
and `saas.api_version`="xxxxx"
and `saas.model`="xxxxx"
and udfName="openai_saas"
and modelTable="command";
```

### zhipu

```sql

!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/zhipu"
and `saas.api_key`="xxxxxxxxxxxxxxxxxx"
and `saas.secret_key`="xxxxxxxxxxxxxxxx"
and `saas.model`="chatglm_lite"
and udfName="zhipu_saas"
and modelTable="command";
```

### minimax

```sql

!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/minimax"
and `saas.api_key`="xxxxxxxxxxxxxxxxxx"
and `saas.group_id`="xxxxxxxxxxxxxxxx"
and `saas.model`="abab5.5-chat"
and `saas.api_url`="https://api.minimax.chat/v1/text/chatcompletion_pro"
and udfName="minimax_saas"
and modelTable="command";

```

### sparkdesk

```sql
!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/sparkdesk"
and `saas.appid`="xxxxxxxxxxxxxxxxxx"
and `saas.api_key`="xxxxxxxxxxxxxxxx"
and `saas.api_secret`="xxxx"
and `gpt_url`="ws://spark-api.xf-yun.com/v1.1/chat"
and udfName="sparkdesk_saas"
and modelTable="command";
```

### baichuan

```sql
!byzerllm setup single;
!byzerllm setup "num_gpus=0";
!byzerllm setup "maxConcurrency=10";

run command as LLM.`` where
action="infer"
and pretrainedModelType="saas/baichuan"
and `saas.api_key`="xxxxxxxxxxxxxxxxxx"
and `saas.secret_key`="xxxxxxxxxxxxxxxx"
and `saas.baichuan_api_url`="https://api.baichuan-ai.com/v1/chat"
and `saas.model`="Baichuan2-53B"
and udfName="baichuan_saas"
and modelTable="command";
```

---

## Pretrain

This section will introduce how to pretrain a LLM model with Byzer-llm.  However, for now, the pretrain feature is more mature in Byzer-SQL, so we will introduce the pretrain feature in Byzer-SQL.

```sql
-- Deepspeed Config
set ds_config='''
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
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
         "device": "cpu"         
     },           
    "offload_param": {
         "device": "cpu"
    },
    "contiguous_gradients": true,
    "allgather_bucket_size": 1e8,
    "reduce_bucket_size": 1e8,
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
''';

-- load data
load text.`file:///home/byzerllm/data/raw_data/*`
where wholetext="true" as trainData;

select value as text,file from trainData  as newTrainData;

-- split the data into 12 partitions
run newTrainData as TableRepartition.`` where partitionNum="12" and partitionCols="file" 
as finalTrainData;


-- setup env, we use 12 gpus to pretrain the model
!byzerllm setup sfft;
!byzerllm setup "num_gpus=12";

-- specify the pretrain model type and the pretrained model path
run command as LLM.`` where 
and localPathPrefix="/home/byzerllm/models/sfft/jobs"
and pretrainedModelType="sfft/llama2"
-- original model is from
and localModelDir="/home/byzerllm/models/Llama-2-7b-chat-hf"
-- and localDataDir="/home/byzerllm/data/raw_data"

-- we use async mode to pretrain the model, since the pretrain process will take several days or weeks
-- Ray Dashboard will show the tensorboard address, and then you can monitor the loss
and detached="true"
and keepPartitionNum="true"

-- use deepspeed config, this is optional
and deepspeedConfig='''${ds_config}'''


-- the pretrain data is from finalTrainData table
and inputTable="finalTrainData"
and outputTable="llama2_cn"
and model="command"
-- some hyper parameters
and `sfft.int.max_length`="128"
and `sfft.bool.setup_nccl_socket_ifname_by_ip`="true"
;
```

Since the deepspeed checkpoint is not compatible with the huggingface checkpoint, we need to convert the deepspeed checkpoint to the huggingface checkpoint. The following code will convert the deepspeed checkpoint to the huggingface checkpoint.

```sql
!byzerllm setup single;

run command as LLM.`` where 
action="convert"
and pretrainedModelType="deepspeed/llama3b"
and modelNameOrPath="/home/byzerllm/models/base_model"
and checkpointDir="/home/byzerllm/data/checkpoints"
and tag="Epoch-1"
and savePath="/home/byzerllm/models/my_3b_test2";
```


Now you can deploy the converted model :

```sql
-- 部署hugginface 模型
!byzerllm setup single;

set node="master";
!byzerllm setup "num_gpus=2";
!byzerllm setup "workerMaxConcurrency=1";

run command as LLM.`` where 
action="infer"
and pretrainedModelType="custom/auto"
and localModelDir="/home/byzerllm/models/my_3b_test2"
and reconnect="false"
and udfName="my_3b_chat"
and modelTable="command";
```

## Finetune

```sql
-- load data, we use the dummy data for finetune
-- data format supported by Byzer-SQL：https://docs.byzer.org/#/byzer-lang/zh-cn/byzer-llm/model-sft

load json.`/tmp/upload/dummy_data.jsonl` where
inferSchema="true"
as sft_data;

-- Fintune Llama2
!byzerllm setup sft;
!byzerllm setup "num_gpus=4";

run command as LLM.`` where 
and localPathPrefix="/home/byzerllm/models/sft/jobs"

-- 指定模型类型
and pretrainedModelType="sft/llama2"

-- 指定模型
and localModelDir="/home/byzerllm/models/Llama-2-7b-chat-hf"
and model="command"

-- 指定微调数据表
and inputTable="sft_data"

-- 输出新模型表
and outputTable="llama2_300"

-- 微调参数
and  detached="true"
and `sft.int.max_seq_length`="512";
```

You can check the finetune actor in the Ray Dashboard, the name of the actor is `sft-william-xxxxx`.

After the finetune actor is finished, you can get the model path, so you can deploy the finetuned model.


Here is the log of the finetune actor:

```
Loading data: /home/byzerllm/projects/sft/jobs/sft-william-20230809-13-04-48-674fd1b9-2fc1-45b9-9d75-7abf07cb84cb/finetune_data/data.jsonl3
2
there are 33 data in dataset
*** starting training ***
{'train_runtime': 19.0203, 'train_samples_per_second': 1.735, 'train_steps_per_second': 0.105, 'train_loss': 3.0778136253356934, 'epoch': 0.97}35

***** train metrics *****36  
epoch                    =       0.9737  
train_loss               =     3.077838  
train_runtime            = 0:00:19.0239  
train_samples_per_second =      1.73540  
train_steps_per_second   =      0.10541

[sft-william] Copy /home/byzerllm/models/Llama-2-7b-chat-hf to /home/byzerllm/projects/sft/jobs/sft-william-20230809-13-04-48-674fd1b9-2fc1-45b9-9d75-7abf07cb84cb/finetune_model/final/pretrained_model4243              
[sft-william] Train Actor is already finished. You can check the model in: /home/byzerllm/projects/sft/jobs/sft-william-20230809-13-04-48-674fd1b9-2fc1-45b9-9d75-7abf07cb84cb/finetune_model/final   
```

You can download the finetuned model from the path `/home/byzerllm/projects/sft/jobs/sft-william-20230809-13-04-48-674fd1b9-2fc1-45b9-9d75-7abf07cb84cb/finetune_model/final`, or copy the model to all other node in the Ray cluster.

Try to deploy the finetuned model:

```sql
!byzerllm setup single;
run command as LLM.`` where 
action="infer"
and localPathPrefix="/home/byzerllm/models/infer/jobs"
and localModelDir="/home/byzerllm/models/sft/jobs/sft-william-llama2-alpaca-data-ccb8fb55-382c-49fb-af04-5cbb3966c4e6/finetune_model/final"
and pretrainedModelType="custom/llama2"
and udfName="fintune_llama2_chat"
and modelTable="command";
```

Byzer-LLM use QLora to finetune the model, you can merge the finetuned model with the original model with the following code:

```sql
-- 合并lora model + base model

!byzerllm setup single;

run command as LLM.`` where 
action="convert"
and pretrainedModelType="deepspeed/llama"
and model_dir="/home/byzerllm/models/sft/jobs/sft-william-20230912-21-50-10-2529bf9f-493e-40a3-b20f-0369bd01d75d/finetune_model/final/pretrained_model"
and checkpoint_dir="/home/byzerllm/models/sft/jobs/sft-william-20230912-21-50-10-2529bf9f-493e-40a3-b20f-0369bd01d75d/finetune_model/final"
and savePath="/home/byzerllm/models/sft/jobs/sft-william-20230912-21-50-10-2529bf9f-493e-40a3-b20f-0369bd01d75d/finetune_model/merge";

```







