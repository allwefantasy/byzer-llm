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
- 0.1.16： Enhance the API for byzer-retrieval
- 0.1.14： add get_tables/get_databases API for byzer-retrieval
- 0.1.13: support shutdown cluster for byzer-retrieval
- 0.1.12: Support Python API (alpha)
- 0.1.5: Support python wrapper for [byzer-retrieval](https://github.com/allwefantasy/byzer-retrieval)

---

## Installation

```bash
pip install -r requirements.txt
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

llm.deploy(model_path="/home/byzerllm/models/openbuddy-llama-13b-v5-fp16",
           pretrained_model_type="custom/llama2",
           udf_name="llama2_chat",infer_params={})

llm.chat("llama2_chat",LLMRequest(instruction="hello world"))[0].output
```

The above code will deploy a llama2 model and then use the model to infer the input text. If you use transformers as the inference backend, you should specify the `pretrained_model_type` manually since the transformers backend can not auto detect the model type.

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

## DeepSpeed Support

The Byzer-llm also support DeepSpeed as the inference backend. The following code will deploy a DeepSpeed model and then use the model to infer the input text.

```python
import ray
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend

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

The following code have the same effect as the above python code.

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

select 
llama2_chat(llm_param(map(
              "user_role","User",
              "assistant_role","Assistant",
              "system_msg",'You are a helpful assistant. Think it over and answer the user question correctly.',
              "instruction",llm_prompt('
Please remenber my name: {0}              
',array("Zhu William"))

)))

 as q as q1;

```

Once you deploy the model with `run command as LLM`, then you can ues the model as a SQL function. This feature is very useful for data scientists who want to use LLM in their data analysis or data engineers who want to use LLM in their data pipeline.

---






