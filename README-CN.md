![logo.jpg](https://raw.gitcode.com/allwefantasy11/byzer-llm/attachment/uploads/f5751555-1419-470c-8a33-dfcdc238789d/logo.jpg 'logo.jpg')

<h3 align="center">
简单、高效且低成本的预训练、微调与服务，惠及大众
</h3>

<p align="center">
| <a href="./README.md"><b>English</b></a> | <a href="./README-CN.md"><b>中文</b></a> |
</p>

---

最新信息🔥

- [2024/01] Release Byzer-LLM 0.1.39
- [2023/12] Release Byzer-LLM 0.1.30

---

Byzer-LLM 基于 Ray 技术构建，是一款覆盖大语言模型（LLM）完整生命周期的解决方案，包括预训练、微调、部署及推理服务等阶段。

Byzer-LLM 的独特之处在于：

1. 全生命周期管理：支持预训练、微调、部署和推理服务全流程
2. 兼容 Python/SQL API 接口
3. 基于 Ray 架构设计，便于轻松扩展

---

* [版本记录](#版本记录) 
* [安装指南](#安装指南) 
* [快速入门](#快速入门) 
* [如何连接来自 Ray 集群外部的模型](#如何连接来自-Ray-集群外部的模型)
* 嵌入/重排序
    * [嵌入模型](#嵌入模型)
    * [嵌入重排序模型](#嵌入重排序模型)
* [量化](#量化) 
* [支持的模型](#支持的模型) 
* 服务端
    * 后端
        * [支持 vLLM ](#支持-vLLM) 
        * [支持 DeepSpeed](#支持-DeepSpeed) 
    * [Byzer-LLM 兼容 OpenAI 的 RESTful API 服务](#兼容-OpenAI-RESTful-API-服务)
* 大语言模型与 Python
    * [函数调用](#函数调用) 
    * [使用 pydantic 类响应](#响应时使用-pydantic-类) 
    * [函数实现](#函数实现功能) 
    * [对 LLM 友好的函数/数据类](#大语言模型友好型函数数据类) 
* 模型配置   
    * [模型元信息](#模型元信息) 
    * [聊天模板](#对话模板) 
    * [LLM 默认参数](#LLM-默认参数) 
* [SaaS 模型](#SaaS-模型) 
    * [通义千问](#通义千问qianwen) 
    * [百川](#百川baichuan)
    * [azure openai](#azure-openai)
    * [openai](#openai)
    * [智谱](#智谱zhipu)
    * [星火](#星火sparkdesk)         
* [多模态](#多模态) 
* [StableDiffusion](#StableDiffusion)
* [SQL 支持](#SQL-支持) 
* [预训练](#预训练)
* [微调](#微调)
* [文章](#文章)
* [贡献指南](#贡献指南)

---

## 版本记录
- 0.1.39：提升函数功能实现 / 更新 SaaS 开发者套件（SDK） / 集成 OpenAI 兼容 API 服务
- 0.1.38：升级 saas/sparkdask 模型组件 / 引入嵌入式重排序模型 / 实现代理消息存储支持
- 0.1.37：对 saas/zhipu 模型进行更新，您可以选用 glm-4 或 embedding-2 用于大语言模型或者嵌入应用场景
- 0.1.36：修正由 Byzer-Agent 更新所导致的数据分析代理模块的故障
- 0.1.35：新增百川 SaaS 嵌入式模型
- 0.1.34：进一步强化 Byzer-Agent API 功能并修复 Byzer-LLM 内部的部分问题
- 0.1.33：解决响应类内部错误 / 新增多项函数实现
- 0.1.32：对 StableDiffusion 进行性能优化
- 0.1.31：启用包含令牌计数信息的实时聊天功能 / 对多模态模型聊天体验进行了优化
- 0.1.30：在 vLLM 后台应用聊天模板功能
- 0.1.29：提升了 DataAnalysis 代理的功能表现
- 00.1.28：修复若干已知 bug
- 0.1.27：修复若干已知 bug
- 0.1.26：支持 QianWen SaaS 平台 / 实现实时聊天功能在 QianWenSaas 中的应用 / 解决部分 SaaS 模型存在的问题
- 0.1.24：支持从模型实例直接提取元数据并自动配置模板
- 0.1.23：通过 Python API 进行模型微调 / 解决了一些现有问题
- 0.1.22：增添了函数调用支持 / 响应结构采用 pydantic 类型定义
- 0.1.19：修复了嵌入相关问题
- 0.1.18：实现了流式聊天功能 / 加入了模型模板支持
- 0.1.17：此版本未有实质性更新内容
- 0.1.16：增强了针对 byzer-retrieval 的 API 功能
- 0.1.14：为 byzer-retrieval 添加了获取表格(get_tables)和数据库(get_databases)的 API 接口
- 0.1.13：支持 byzer-retrieval 能够关闭集群操作
- 0.1.12：初步支持 Python API（尚处于 alpha 测试阶段）
- 0.1.5：支持 Python 封装形式的 [byzer-retrieval](https://github.com/allwefantasy/byzer-retrieval)

---


## 安装指南

推荐配置环境:

1. Conda:  python==3.10.11  
2. OS:     ubuntu 22.04
3. Cuda:   12.1.0 (可选，仅在您使用SaaS模型时使用)

```bash
## Make sure you python version is 3.10.11
pip install -r requirements.txt
## Skip this step if you have no Nvidia GPU
pip install vllm==0.2.6
pip install -U byzerllm
ray start --head
```

若你的 CUDA 版本为 11.8，请参照以下链接来安装 vLLM：
https://docs.vllm.ai/en/latest/getting_started/installation.html

安装过程中需关注的关键环节如下：


```shell
As of now, vLLM’s binaries are compiled on CUDA 12.1 by default. However, you can install vLLM with CUDA 11.8 by running:

# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.2.6
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl

# Re-install PyTorch with CUDA 11.8.
pip uninstall torch -y
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118

# Re-install xFormers with CUDA 11.8.
pip uninstall xformers -y
pip install --upgrade xformers --index-url https://download.pytorch.org/whl/cu118
```

### 原始机器配置指南

> 本方案已针对 Ubuntu 20.04/22.04 版本和 CentOS 8.0 操作系统完成测试

若您手头的计算机尚处于初始状态，即未安装 GPU 驱动和 CUDA 环境，可按以下提供的脚本步骤轻松完成机器配置：

```shell
git clone https://gitee.com/allwefantasy/byzer-llm
cd byzer-llm/setup-machine
```

接下来，请切换至 **ROOT**，并执行以下准备好的自动化配置脚本：

```shell
ROLE=master ./setup-machine.sh
```
紧接着，系统将为您新建一个名为 `byzerllm` 的用户账户。

随后，请切换至这个新建的 `byzerllm` 用户身份，并执行以下配置脚本：

```shell
ROLE=master ./setup-machine.sh
```

脚本会自动为您安装以下各项软件：

1. 
2. Conda
3. Nvidia Driver 535
4. Cuda 12.1.0
5. Ray 
6. requirements.txt 文件内所列举的所有 Python 第三方库
7. Byzer-SQL/Byzer-Notebook 大数据处理与分析工具

若您需要向 Ray 集群扩展更多工作节点，只需在新增的工作节点上重复以上安装步骤。
请注意，在工作节点上，`ROLE` 应为 `worker`。

```shell
ROLE=worker ./setup-machine.sh
```

---

## 快速入门

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



llm_client = ByzerLLM()
llm_client.setup_template("llama2_chat","auto")

v = llm.chat_oai(model="llama2_chat",conversations=[{
    "role":"user",
    "content":"hello",
}])

print(v[0].output)
```

上述代码将会加载并部署一个名为 llama2 的模型，然后利用该模型对输入文本进行推理分析。如果你选择 transformers 作为推理后端引擎，需要注意的是需要手动设定 `pretrained_model_type` 参数，因为 transformers 本身不具备自动检测模型类型的机能。

Byzer-LLM 同样支持以相同方式调用并部署云端（SaaS）模型。这一功能为开源模型和云服务模型提供了一个统一的操作界面。接下来的示例代码将展示如何部署来自 Azure OpenAI 的模型，并在其后利用这个模型对输入文本进行推理处理。


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


llm_client = ByzerLLM()
llm_client.setup_template("azure_openai","auto")

v = llm.chat_oai(model="azure_openai",conversations=[{
    "role":"user",
    "content":"hello",
}])

print(v[0].output)
```

请注意，鉴于 SaaS 模型无需依赖 GPU，我们把 `setup_gpus_per_worker` 参数设为 0。另外，你可以借助 `setup_num_workers` 参数来调整最大并发执行数，然而要注意的是，SaaS 模型自带其并发请求的上限，因此 `setup_num_workers` 参数所控制的是 Byzer-LLM 接受的最大并发任务数，而非绝对的并发执行上限，实际并发执行数仍需参照 SaaS 模型自身的并发限制。

## 如何连接来自 Ray 集群外部的模型

建议的最佳实践是在您的目标设备（例如 Web 服务器）上启动一个闲置的 Ray 工作节点：

```shell
ray start --address="xxxxx:6379"  --num-gpus=0 --num-cpus=0 
```

这样一来，您便可以从 Ray 集群外部顺利对接所需模型：

```python
import ray
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend

## connect the ray cluster by the empty worker we started before
## this code should be run once in your prorgram
ray.init(address="auto",namespace="default",ignore_reinit_error=True)

## new a ByzerLLM instance

llm_client = ByzerLLM()
llm_client.setup_template("llama2_chat","auto")

v = llm.chat_oai(model="llama2_chat",conversations=[{
    "role":"user",
    "content":"hello",
}])

print(v[0].output)
```


## 嵌入模型

以下展示的代码片段是一个关于部署 BGE 嵌入模型的实际案例

```python
import ray
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend
ray.init(address="auto",namespace="default",ignore_reinit_error=True)
llm = ByzerLLM()

llm.setup_gpus_per_worker(0.4).setup_num_workers(2).setup_infer_backend(InferBackend.Transformers)
llm.deploy(
    model_path="/home/byzerllm/models/bge-large-zh",
    pretrained_model_type="custom/bge",
    udf_name="emb",
    infer_params={}
)   
```

这样，您就能够将任意一段文本成功转化为向量表示：

```python
t = llm.emb("emb",LLMRequest(instruction="wow"))
t[0].output
#output: [-0.005588463973253965,
 -0.01747054047882557,
 -0.040633779019117355,
...
 -0.010880181565880775,
 -0.01713103987276554,
 0.017675869166851044,
 -0.010260719805955887,
 ...]
```

Byzer-LLM 还支持云端（SaaS）嵌入模型服务。下面这段代码演示了如何部署一个百川提供的嵌入模型，并利用该模型对输入的文本进行向量化处理。

```python
import os
os.environ["RAY_DEDUP_LOGS"] = "0" 

import ray
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.client import ByzerLLM,LLMRequest,LLMResponse,LLMHistoryItem,InferBackend
from byzerllm.utils.client import Templates

ray.init(address="auto",namespace="default",ignore_reinit_error=True)  

llm = ByzerLLM(verbose=True)

llm.setup_num_workers(1).setup_gpus_per_worker(0)

chat_name = "baichuan_emb"
if llm.is_model_exist(chat_name):
    llm.undeploy(udf_name=chat_name)

llm.deploy(model_path="",
           pretrained_model_type="saas/baichuan",
           udf_name=chat_name,
           infer_params={
            "saas.api_key":"",            
            "saas.model":"Baichuan-Text-Embedding"
           })
llm.setup_default_emb_model_name(chat_name)

v = llm.emb(None,LLMRequest(instruction="你好"))
print(v.output)
```

## 嵌入重排序模型

若您打算利用嵌入重排序模型进行优化，可以参考以下具体应用示例。

```python
import ray
from byzerllm.utils.client import ByzerLLM,LLMRequest,InferBackend
ray.init(address="auto",namespace="default",ignore_reinit_error=True)
llm = ByzerLLM()

llm.setup_gpus_per_worker(0.4).setup_num_workers(2).setup_infer_backend(InferBackend.Transformers)
llm.deploy(
    model_path="/Users/wanghan/data/bge-reranker-base",
    pretrained_model_type="custom/bge_rerank",
    udf_name="emb_rerank",
    infer_params={}
)   
llm.setup_default_emb_model_name("emb_rerank")
```
接下来，您可以通过将查询文本和待评估文本送入重排序模型，得到它们之间的相关性得分。

```python
sentence_pairs_01 = ['query', 'passage']
t1 = llm.emb_rerank(sentence_pairs=sentence_pairs_01)
print(t1[0].output)
#output [['query', 'passage'], 0.4474925994873047]

sentence_pairs_02 = [['what is panda?', 'hi'], ['what is panda?','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
t2 = llm.emb_rerank(sentence_pairs=sentence_pairs_02)
print(t2[0].output)
#output [[['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.'], 6.1821160316467285], [['what is panda?', 'hi'], -8.154398918151855]]
```

## 量化

当后端采用 `InferBackend.transformers` 时，这里展示的是一个关于“百川2”模型的实例应用。

```python
llm.setup_gpus_per_worker(2).setup_num_workers(1).setup_infer_backend(InferBackend.Transformers)
llm.deploy(
    model_path=model_location,
    pretrained_model_type="custom/baichuan2",
    udf_name="baichuan2_13_chat",
    infer_params={"quatization":"4"}
)
```
目前支持的 `quantization`（量化）选项包括：

1. 4
2. 8
3. true/false

若将该参数设为 true，系统将采用 int4 量化级别。

针对后端为 `InferBackend.VLLM` 的情况，以下是一个使用“易”模型的示例：

若需要部署经过量化压缩的模型，您可以按照以下代码样式设置 `infer_params` 参数：

```python
llm.setup_gpus_per_worker(1).setup_num_workers(1).setup_infer_backend(InferBackend.VLLM)
llm.deploy(
    model_path="/home/winubuntu/models/Yi-6B-Chat-4bits",
    pretrained_model_type="custom/auto",
    udf_name="chat",
    infer_params={"backend.quantization":"AWQ"}
)
```

`backend.quantization` 参数可以选用 GPTQ 或 AWQ 两种量化方法。


## 支持的模型列表

支持的开源 `pretrained_model_type` 包括：

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

支持的 SaaS `pretrained_model_type` 如下：

1. saas/chatglm	Chatglm130B
2. saas/sparkdesk	星火大模型
3. saas/baichuan	百川大模型
4. saas/zhipu	智谱大模型
5. saas/minimax	MiniMax 大模型
6. saas/qianfan	文心一言
7. saas/azure_openai	
8. saas/openai

请注意，源自 lama/llama2/starcode 的衍生模型也同样受到支持。例如，您可以使用 `llama` 加载 vicuna 模型。

## 支持 vLLM

Byzer-LLM 同样具备支持将 vLLM 作为推理后端的能力。这意味着您可以依据以下代码范例，部署一个 vLLM（虚拟大规模语言模型），并借此模型对给定文本进行智能推理处理。

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
    infer_params={}
)

v = llm.chat_oai(model="zephyr_chat",conversations=[{
    "role":"user",
    "content":"hello",
}])
print(v[0].output)
```

vLLM 与 transformers 后端在使用上有一些微小的不同点：

1. 在 vLLM 中，`pretrained_model_type` 参数固定为 `custom/auto`，这是因为 vLLM 自带模型类型自动检测功能。
2. 若要指定推理后端为 vLLM，请将 `setup_infer_backend` 参数设置为 `InferBackend.VLLM`。
 

### 流式对话

若模型采用了 vLLM 后端进行部署，它还将支持“流式对话”特性：

调用 `stream_chat_oai` 方法可以获得一个生成器，进而逐条拉取模型生成的回复文本。

```python

llm.setup_default_model_name(chat_model_name) 

t = llm.stream_chat_oai(conversations=[{
    "role":"user",
    "content":"Hello, how are you?"
}])

for line in t:
   print(line+"\n")
```

## 支持 DeepSpeed

Byzer-LLM 还支持将 DeepSpeed 作为模型推理的后端技术。以下代码片段将展示如何部署 DeepSpeed 优化的模型，并利用该模型对输入文本进行推理分析：

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

上述代码与用于 vLLM 的代码基本一致，唯一的区别在于 `InferBackend` 设置成了 `InferBackend.DeepSpeed`。

## 兼容 OpenAI RESTful API 服务

通过执行下列代码片段，即可启动一个能够与 OpenAI 对接的 ByzerLLm 大语言模型 RESTful API 服务器：

```shell
ray start --address="xxxxx:6379"  --num-gpus=0 --num-cpus=0 
python -m byzerllm.utils.client.entrypoints.openai.api_server
```

默认情况下，服务器运行时会在`8000`端口等待请求。您可以采用如下代码片段来验证并测试该 API 功能：

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="xxxx"
)

chat_completion = client.chat.completions.create(    
    model="wenxin_chat",
    messages=[{"role": "user", "content": "写一个排序算法"}],
    stream=False
)

print(chat_completion.choices[0].message.content)
```

## 流式对话

```python

from openai import OpenAI
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="simple"
)

chat_completion = client.chat.completions.create(    
    model="wenxin_chat",
    messages=[{"role": "user", "content": "写一个排序算法"}],
    stream=True
)

for chunk in chat_completion:    
    print(chunk.choices[0].delta.content or "", end="")
```

## 函数调用

这有一个利用 QWen 72B 模型进行函数调用的基础示例。

部署模型的步骤演示：

```python
import ray
ray.init(address="auto",namespace="default") 
llm = ByzerLLM()

model_location="/home/byzerllm/models/Qwen-72B-Chat"

llm.setup_gpus_per_worker(8).setup_num_workers(1).setup_infer_backend(InferBackend.VLLM)
llm.deploy(
    model_path=model_location,
    pretrained_model_type="custom/auto",
    udf_name=chat_model_name,
    infer_params={}
)

llm.setup_default_model_name("chat")
# from 0.1.24 
# llm.setup_auto("chat")
meta = llm.get_meta()
llm.setup_max_model_length("chat",meta.get("max_model_len",32000))
lm.setup_template("chat",Templates.qwen()) 
```

让我们一起尝试编写几个Python函数，先来体验如何使用QWen 72B模型生成回复：

```python

from typing import List,Dict,Any,Annotated
import pydantic 
import datetime
from dateutil.relativedelta import relativedelta

def compute_date_range(count:Annotated[int,"时间跨度，数值类型"],
                       unit:Annotated[str,"时间单位，字符串类型",{"enum":["day","week","month","year"]}])->List[str]:
    '''
    计算日期范围

    Args:
        count: 时间跨度，数值类型
        unit: 时间单位，字符串类型，可选值为 day,week,month,year
    '''        
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    if unit == "day":
        return [(now - relativedelta(days=count)).strftime("%Y-%m-%d %H:%M:%S"),now_str]
    elif unit == "week":
        return [(now - relativedelta(weeks=count)).strftime("%Y-%m-%d %H:%M:%S"),now_str]
    elif unit == "month":
        return [(now - relativedelta(months=count)).strftime("%Y-%m-%d %H:%M:%S"),now_str]
    elif unit == "year":
        return [(now - relativedelta(years=count)).strftime("%Y-%m-%d %H:%M:%S"),now_str]
    return ["",""]

def compute_now()->str:
    '''
    计算当前时间
    '''
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

这里我们给出了两个便捷的函数：

1. compute_date_range：根据用户指定的数量（如天数、周数等）和单位来计算一个起止日期的区间。
2. compute_now：获取当前的日期信息。

当面对用户的具体问题时，我们会利用模型调用这两个功能工具。

```python
t = llm.chat_oai([{
    "content":'''计算当前时间''',
    "role":"user"    
}],tools=[compute_date_range,compute_now],execute_tool=True)

t[0].values

## output: ['2023-12-18 17:30:49']
```

```python
t = llm.chat_oai([{
    "content":'''最近三个月趋势''',
    "role":"user"    
}],tools=[compute_date_range,compute_now],execute_tool=True)

t[0].values

## output: [['2023-09-18 17:31:21', '2023-12-18 17:31:21']]
```

```python
t = llm.chat_oai([{
    "content":'''最近三天''',
    "role":"user"    
}],tools=[compute_date_range,compute_now],execute_tool=True)

t[0].values

## output: [['2023-12-15 17:23:38', '2023-12-18 17:23:38']]
```

```python
t = llm.chat_oai([{
    "content":'''你吃饭了么？''',
    "role":"user"    
}],tools=[compute_date_range,compute_now],execute_tool=True)

if t[0].values:
    print(t[0].values[0])
else:
    print(t[0].response.output)   

## output: '您好，我是一个人工智能语言模型，暂时无法吃饭。'
```

您可以检查 `from byzerllm.utils import function_calling_format` 中的默认提示模板函数。如果模型使用默认函数效果不佳，您可以设置自定义函数：

```python
def custom_function_calling_format(prompt:str,tools:List[Callable],tool_choice:Callable)->str:
.....


llm.setup_function_calling_format_func("chat",custom_function_calling_format)
```

## 响应时使用 Pydantic 类

在与大语言模型交谈时，你可以自定义设置一个类似“响应类”（Response Class）的结构，以此来规范和控制模型给出回答的数据格式和结构

```python
import pydantic 

class Story(pydantic.BaseModel):
    '''
    故事
    '''

    title: str = pydantic.Field(description="故事的标题")
    body: str = pydantic.Field(description="故事主体")



t = llm.chat_oai([
{
    "content":f'''请给我讲个故事，分成两个部分，一个标题，一个故事主体''',
    "role":"user"
},
],response_class=Story)

t[0].value

## output: Story(title='勇敢的小兔子', body='在一个美丽的森林里，住着一只可爱的小兔子。小兔子非常勇敢，有一天，森林里的动物们都被大灰狼吓坏了。只有小兔子站出来，用智慧和勇气打败了大灰狼，保护了所有的动物。从此，小兔子成为了森林里的英雄。')
```

上述代码会让 LLM 直接生成 Story 类的对象。但在某些情况下，我们希望 LLM 先生成文本，再从文本中提取结构信息，这时可以通过设置 `response_after_chat=True` 来启用这一行为。不过，请注意，这样做会导致一定的性能损耗（额外的推理计算）。

```python
t = llm.chat_oai([
{
    "content":f'''请给我讲个故事，分成两个部分，一个标题，一个故事主体''',
    "role":"user"
},
],response_class=Story,response_after_chat=True)

t[0].value
## output: Story(title='月光下的守护者', body='在一个遥远的古老村庄里，住着一位名叫阿明的年轻人。阿明是个孤儿，从小在村里长大，以种田为生。他善良、勤劳，深受村民们喜爱。\n\n村子里有个传说，每当满月时分，月亮女神会在村子后山的古树下出现，赐福给那些善良的人们。然而，只有最纯洁的心才能看到她。因此，每年的这个时候，阿明都会独自一人前往后山，希望能得到女神的祝福。\n\n这一年，村子遭受了严重的旱灾，庄稼枯黄，人们生活困苦。阿明决定向月亮女神祈求降雨，拯救村子。他在月光下虔诚地祈祷，希望女神能听到他的呼唤。\n\n就在这个时刻，月亮女神出现了。她被阿明的善良和执着所感动，答应了他的请求。第二天早晨，天空乌云密布，大雨倾盆而下，久旱的土地得到了滋润，庄稼重新焕发生机。\n\n从此以后，每年的满月之夜，阿明都会去后山等待月亮女神的出现，他成为了村民心中的守护者，用他的善良和执着，守护着整个村庄。而他也终于明白，真正的守护者，并非需要超凡的力量，只需要一颗充满爱与善良的心。')
```

你可以在 byzerllm.utils 模块中通过 import 语句引入默认的提示模板函数，`from byzerllm.utils import response_class_format,response_class_format_after_chat`。

如果模型使用默认函数的效果不尽如人意，你可以设置自定义函数来优化它：

```python
def custom_response_class_format(prompt:str,cls:pydantic.BaseModel)->str:
.....


llm.setup_response_class_format_func("chat",custom_response_class_format)
```

## 函数实现功能

Byzer-LLM 还支持函数实现功能。您可以定义一个空函数，并结合函数内的文档说明/用户提出的问题，来引导大语言模型(LLM)去实现这个函数的功能。

下面是一个简单的示例：

```python
from byzerllm.utils.client import code_utils,message_utils
from typing import List,Union,Optional
import pydantic

class Time(pydantic.BaseModel):
    time: str = pydantic.Field(...,description="时间，时间格式为 yyyy-MM-dd")


@llm.impl()
def calculate_current_time()->Time:
    '''
    计算当前时间
    '''
    pass 


calculate_current_time()
#output: Time(time='2024-01-28')
```

默认情况下，系统会把函数内部的计算过程（即函数实现）缓存起来，这样当下次调用相同函数时就能迅速执行，无需重新计算。

```python
start = time.monotonic()
calculate_current_time()
print(f"first time cost: {time.monotonic()-start}")

start = time.monotonic()
calculate_current_time()
print(f"second time cost: {time.monotonic()-start}")

# output:
# first time cost: 6.067266260739416
# second time cost: 4.347506910562515e-05
```

若要清除缓存，可以通过运行 `llm.clear_impl_cache()` 方法来实现这一目的。

接下来是一个展示如何针对带参数的函数执行结果进行缓存处理的示例：

```python
from byzerllm.utils.client import code_utils,message_utils
from typing import List,Union,Optional,Annotated
import pydantic
from datetime import datetime

class Time(pydantic.BaseModel):
    time: str = pydantic.Field(...,description="时间，时间格式为 yyyy-MM-dd")


@llm.impl()
def add_one_day(current_day:Annotated[datetime,"当前日期，类型是datatime.datetime"])->Time:
    '''
    给传入的日期加一天，得到明天的时间
    '''
    pass 


add_one_day(datetime.now())
# output:Time(time='2024-01-29')
```

操作指引：

```python
from byzerllm.utils.client import code_utils,message_utils
from typing import List,Union,Optional
import pydantic

class TimeRange(pydantic.BaseModel):
    '''
    时间区间
    格式需要如下： yyyy-MM-dd
    '''  
    
    start: str = pydantic.Field(...,description="开始时间.时间格式为 yyyy-MM-dd")
    end: str = pydantic.Field(...,description="截止时间.时间格式为 yyyy-MM-dd")

@llm.impl(instruction="去年三月到七月")
def calculate_time_range()->TimeRange:
    '''
    计算时间区间，时间格式为 yyyy-MM-dd. 
    '''
    pass 

calculate_time_range()
# output: TimeRange(start='2023-03-01', end='2023-07-31')
```

若想将用户的查询问题用于替代原先用来清除缓存的指令，可以采用如下代码实现这一功能：

```python
from byzerllm.utils.client import code_utils,message_utils
from typing import List,Union,Optional
import pydantic

class TimeRange(pydantic.BaseModel):
    '''
    时间区间
    格式需要如下： yyyy-MM-dd
    '''  
    
    start: str = pydantic.Field(...,description="开始时间.时间格式为 yyyy-MM-dd")
    end: str = pydantic.Field(...,description="截止时间.时间格式为 yyyy-MM-dd")

def calculate_time_range()->TimeRange:
    '''
    计算时间区间，时间格式为 yyyy-MM-dd. 
    '''
    pass 


llm.impl(instruction="去年三月到七月")(calculate_time_range)()
```

若想深入了解函数实现的详细情况，可在调用时加上 `verbose=True` 参数，系统将为你提供更多相关信息：

```python
@llm.impl()
def add_one_day(current_day:Annotated[datetime,"当前日期，类型是datatime.datetime"])->Time:
    '''
    给传入的日期加一天，得到明天的时间
    '''
    pass 
```

你也可以使用基础的 chat_oai 函数来实现函数：

```python
class TimeRange(pydantic.BaseModel):
    '''
    时间区间
    格式需要如下： yyyy-MM-dd
    '''  
    
    start: str = pydantic.Field(...,description="开始时间.时间格式为 yyyy-MM-dd")
    end: str = pydantic.Field(...,description="截止时间.时间格式为 yyyy-MM-dd")


def calculate_time_range():
    '''
    计算时间区间，时间格式为 yyyy-MM-dd. 
    '''
    pass 
    
t = llm.chat_oai([{
    "content":"去年三月到七月",
    "role":"user"    
}],impl_func=calculate_time_range,response_class=TimeRange,execute_impl_func=True)
```

上述代码中，我们定义了一个名为 `calculate_time_range` 的函数，该函数目前为空。接着我们在文档字符串中详细描述了函数的功能，并定义了响应类 `TimeRange`，确保函数返回一个 `TimeRange` 实例。由于该函数应服务于解答用户的问题，所以它的实现应当与用户的具体问题紧密相关。我们不是要去实现一个通用的函数，而是实现一个专门针对用户当前问题进行解答的函数。

执行后，你会得到如下所示的输出结果：

```python
t[0].value
# start='2023-03-01' end='2023-07-31'
```

如果返回的值是 None 或不正确，系统将会给出错误提示信息：
```python
t[0].metadata.get("resason","")
```

如果你定义的函数带有参数，可以通过 `impl_func_params` 参数传递给该函数：

```python
t = llm.chat_oai([{
    "content":"xxxxx",
    "role":"user"    
}],
impl_func=calculate_time_range,
impl_func_params={},
response_class=TimeRange,execute_impl_func=True)
```

如果你想要替换默认的提示模板函数，这里有一个示例：

```python
import pydantic
from typing import List,Optional,Union,Callable
from byzerllm.utils import serialize_function_to_json

def function_impl_format2(prompt:str,func:Optional[Union[Callable,str]],
                             cls:Union[pydantic.BaseModel,str])->str:
    
    tool_choice_ser = serialize_function_to_json(func)    
    _cls = ""
    if isinstance(cls, str):
        _cls = cls
    else:
        _cls = cls.schema_json(ensure_ascii=False)
    
    msg = f''''生成一个python函数，给出详细的思考逻辑，对最后生成的函数不要进行示例说明。

生成的函数的名字以及参数需要满足如下约束：

\```json
{tool_choice_ser}
\```

生成的函数的返回值必须是 Json 格式。

下面是使用 OpenAPI 3.1. 规范描述了你需如何进行json格式的生成。

\```json
{_cls}
\```

根据用的户问题,{func.__doc__}。用户的问题是：{prompt}

请你实现这个函数。
''' 
    
    return msg

llm.setup_impl_func_format_func(chat_model_name,function_impl_format2)
```
默认的提示模板函数是 `function_impl_format`，你可以在 `from byzerllm.utils import function_impl_format` 这段代码中查看其源代码。

## 大语言模型友好型函数/数据类

若要提升函数调用或响应类的性能表现，应当确保你的函数（工具）和数据类对大语言模型（LLM）友好。

接下来，我们一起来看一段 Python 代码示例：

```python
def compute_date_range(count:int, unit:str)->List[str]:                   
    now = datetime.datetime.now()
    ....
```

这段代码并非对大语言模型友好，因为它难以让人或LLM理解函数的具体用途以及输入参数的含义。

大语言模型就如同人类一样，很难让它知道何时或如何调用这个函数。特别是参数`unit`实际上是一个枚举值，但是大语言模型无法获知这一信息。

因此，为了让大语言模型更好地理解 Byzer-LLM 中的这个函数，你应该遵循以下要求：

1. 添加符合 Python 规范的函数注释
2. 使用类型注解为每个参数提供类型和注释，如果参数是一个枚举值，还需要提供枚举的所有可能取值。

下面是改进后的对大语言模型友好的函数定义示例

```python
def compute_date_range(count:Annotated[int,"时间跨度，数值类型"],
                       unit:Annotated[str,"时间单位，字符串类型",{"enum":["day","week","month","year"]}])->List[str]:
    '''
    计算日期范围

    Args:
        count: 时间跨度，数值类型
        unit: 时间单位，字符串类型，可选值为 day,week,month,year
    '''        
    now = datetime.datetime.now()
    ....
```

如果大语言模型（LLM）在调用你的函数时出现问题（例如提供了错误的参数），试着优化函数注释和参数的类型注解注释，以帮助LLM更好地理解函数的正确用法和参数含义。

## 模型元信息

Byzer-LLM 同样支持获取模型实例的元信息。下面的代码将获取名为 `chat` 的模型实例的元信息：

```python
llm.get_meta(model="chat")

#output:
# {'model_deploy_type': 'proprietary',
#  'backend': 'ray/vllm',
#  'max_model_len': 32768,
#  'architectures': ['QWenLMHeadModel']}
```

## 对话模板

不同的模型拥有各自的对话模板，Byzer-LLM 为各个模型提供了一些预设的对话模板。你可以通过以下代码来设置对话模板：

```python
from byzerllm.utils.client import Templates
llm.setup_template("chat",Templates.qwen()) 
```

不仅如此，我们还支持使用 `tokenizer.apply_chat_template` 方法，你可以通过以下代码应用对话模板：

```python
llm.setup_template("chat","auto") 
```

要是模型跟 `tokenizer.apply_chat_template` 这个小工具玩不转，它会发出信号——也就是抛出一个异常。这时候，你完全可以亲自出手，用 `llm.setup_template` 这个招式来手动打造聊天模板。

此外，你还能用 `llm.get_meta` 这个探测器去瞧瞧，看看咱家的模型到底支不支持 `apply_chat_template` 这项技能：

```python
llm.get_meta(model="chat")
```

输出：

```json
{'model_deploy_type': 'proprietary',
 'backend': 'ray/vllm',
 'support_stream': True,
 'support_chat_template': True,
 'max_model_len': 4096,
 'architectures': ['LlamaForCausalLM']}
```

注意，这项特性会触发额外的RPC调用，因此会造成一定的性能损失。

## LLM 默认参数

Byzer-LLM 同样支持为模型设置默认生成参数。以下代码将为名为 `chat` 的模型实例设置默认生成参数：

```python
llm.setup_extra_generation_params("chat",{
    "generation.stop_token_ids":[7]
})
```

在这个例子中，对于模型实例 `chat`，我们将把 `generation.stop_token_ids` 参数设置为数组 `[7]`。这意味着每次调用 `chat` 模型执行文本生成任务时，系统会自动使用这个预设值，即将停止生成序列的标识符`generation.stop_token_ids`设为 `[7]` 的特殊标记。当模型在生成过程中遇到该停用词ID时，就会停止生成新的文本片段。

## 多模态

Byzer 大语言模型（Byzer-LLM）同样具备处理多模态数据的能力。接下来展示的代码片段将会部署一个多模态模型，随后运用此模型对输入文本进行智能推断。

```python
import ray
from byzerllm.utils.client import ByzerLLM,InferBackend

ray.init(address="auto",namespace="default")   

llm = ByzerLLM()
chat_model_name = "qwen_vl_chat"
model_location = "/home/byzerllm/models/Qwen-VL-Chat"

llm.setup_gpus_per_worker(1).setup_num_workers(1).setup_infer_backend(InferBackend.Transformers)
llm.deploy(
    model_path=model_location,
    pretrained_model_type="custom/qwen_vl_chat",
    udf_name=chat_model_name,
    infer_params={}
)    
```

随后，你可以借助这个模型进行实时对话交互：

```python
import base64
image_path = "/home/byzerllm/projects/jupyter-workspace/1.jpg"
with open(image_path, "rb") as f:
    image_content = base64.b64encode(f.read()).decode("utf-8")

t = llm.chat_oai(conversations=[{
    "role": "user",
    "content": "这是什么"
}],model=chat_model_name,llm_config={"image":image_content})

t[0].output

# '{"response": "图中是一名女子在沙滩上和狗玩耍，旁边的狗是一只拉布拉多犬，它坐在沙滩上，面对着一名身穿格子衬衫的女子。女子的腿有些残疾，但是她依然坚持坐在沙滩上和狗玩耍。她的右手拿着一个小玩具，这个玩具上面有两行黑色字母，具体是什么内容看不清楚。她打算把玩具扔给拉布拉多犬。", "history": [{"role": "user", "content": "Picture 1: <img>/tmp/byzerllm/visualglm/images/23eb4cea-cb6e-4f55-8adf-3179ca92ab42.jpg</img>\\n这是什么"}, {"role": "assistant", "content": "图中是一名女子在沙滩上和狗玩耍，旁边的狗是一只拉布拉多犬，它坐在沙滩上，面对着一名身穿格子衬衫的女子。女子的腿有些残疾，但是她依然坚持坐在沙滩上和狗玩耍。她的右手拿着一个小玩具，这个玩具上面有两行黑色字母，具体是什么内容看不清楚。她打算把玩具扔给拉布拉多犬。"}]}'
```

接下来这段代码可以帮助你连续不断地与模型进行多回合的对话交流：

```python
import json
history = json.loads(t[0].output)["history"]

llm.chat_oai(conversations=history+[{
    "role": "user",
    "content": "能圈出狗么？"
}],model=chat_model_name,llm_config={"image":image_content})

# [LLMResponse(output='{"response": "<ref>狗</ref><box>(221,425),(511,889)</box>", "history": [{"role"
```

首先，提取上次对话的聊天记录，然后将这部分历史内容融入新的对话环节，进而继续开展新的对话交流。

## StableDiffusion

Tyzer 大语言模型（Byzer-LLM）同样支持集成 StableDiffusion 技术作为其底层推理框架。接下来的代码将部署一个基于 StableDiffusion 的模型，并借助此模型对输入文本进行深度理解和视觉生成等方面的智能推断。

```python
import ray
from byzerllm.utils.client import ByzerLLM,InferBackend

ray.init(address="auto",namespace="default")   

llm = ByzerLLM()
chat_model_name = "sd_chat"
model_location = "/home/byzerllm/models/stable-diffusion-v1-5"

llm.setup_gpus_per_worker(2).setup_num_workers(1).setup_infer_backend(InferBackend.Transformers)
llm.deploy(
    model_path=model_location,
    pretrained_model_type="custom/stable_diffusion",
    udf_name=chat_model_name,
    infer_params={}
)

def show_image(content):
    from IPython.display import display, Image
    import base64             
    img = Image(base64.b64decode(content))
    display(img)    
    
```

然后就可以通过这个模型进行对话：

```python
import json
t = llm.chat_oai(
    conversations=[
        {
            "role":"user",
            "content":"画一只猫"
        }
    ],model=chat_model_name,llm_config={"gen.batch_size":3}
)

cats = json.loads(t[0].output)
for res in cats:
    show_image(res["img64"])
```

输出：

![](./images/cat2.png)

The parameters:
参数配置：

| 参数                        | 含义                                                         | 默认值   |
| --------------------------- | ------------------------------------------------------------ | -------- |
| Instruction                 | prompt                                                       | 非空     |
| generation.negative_prompt  | 反向的prompt                                                 | ""       |
| generation.sampler_name     | 调度名(unpic, euler_a,euler,ddim,ddpm,deis,dpm2,dpm2-a,dpm++_2m,dpm++_2m_karras,heun,heun_karras,lms,pndm:w) | euler_a  |
| generation.sampling_steps   | 生成的步骤数                                                 | 25       |
| generation.batch_size       | 一次生成几张                                                 | 1        |
| generation.batch_count      | 生成几次                                                     | 1        |
| generation.cfg_scale        | 随机或贴合程度值,值越小生成的图片离你的Tags描述的内容差距越大 | 7.5      |
| generation.seed             | 随机种子                                                     | -1       |
| generation.width            | 图片宽度                                                     | 768      |
| generation.height           | 图片高度                                                     | 768      |
| generation.enable_hires     | 开启高分辨率修复功能(和下面两个一组)                         | false    |
| generation.upscaler_mode    | 放大算法(bilinear, bilinear-antialiased,bicubic,bicubic-antialiased,nearest,nearest-exact) | bilinear |
| generation.scale_slider     | 放大比例                                                     | 1.5      |
| generation.enable_multidiff | 图片分割处理(减少显存销耗)(和下面3个一组)                    | false    |
| generation.views_batch_size | 分批处理规模                                                 | 4        |
| generation.window_size      | 切割大小，宽，高                                             | 64       |
| generation.stride           | 步长                                                         | 16       |
| generation.init_image       | 初始化图片，基于这个图片处理(必须传输base64加密的图片) (和下面的一组) | None     |
| generation.strength         | 重绘幅度: 图像模仿自由度，越高越自由发挥，越低和参考图像越接近，通常小于0.3基本就是加滤镜 | 0.5      |



## SQL 支持

除了 Python 接口之外，Byzer-llm 同样兼容 SQL API。若要使用 SQL API 功能，请先确保安装 Byzer-SQL 语言。

可采用如下命令来安装 Byzer-SQL 语言：

```bash
git clone https://gitee.com/allwefantasy/byzer-llm
cd byzer-llm/setup-machine
sudo -i 
ROLE=master ./setup-machine.sh
```

安装成功后，您可以访问本地 Byzer 控制台，地址为：`http://localhost:9002`。

在 Byzer 控制台内，您可以执行如下 SQL 命令来部署 llama2 模型，该模型的功能与前述 Python 代码片段完全一致。

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

接下来，您可以通过调用名为 `llama2_chat` 的 UDF（用户自定义函数）来激活和使用这个lama2模型：

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

当你使用类似 `run command as LLM` 的方式部署了模型后，就能够如同调用 SQL 函数那样来使用该模型。这一特点极大地便利了那些希望建立数据分析模型时融入大语言模型能力的数据科学家，以及期望在构建数据处理流水线时利用大语言模型功能的数据工程师们。

---

### QWen 

在 ByzerLLM 中使用 QWen 功能时，你需要手动设置几个关键参数：

1. 角色对应关系（角色映射）
2. 终止标识符列表（结束符号ID列表）
3. 从生成的回答中去除终止标识符（简单地说，就是在生成结果中裁剪掉代表对话结束的特殊符号）

为了方便大家操作，我们提供了一个预设模板，你可以试试下面这段代码：

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
## SaaS 模型

鉴于各类 SaaS 模式具有各自的定制参数，这里我们为您准备了一系列 SaaS 模型部署所需的模板，助力您轻松完成不同 SaaS 模型的部署工作。

### 百川（baichuan）

```python

import ray
from byzerllm.utils.client import ByzerLLM

ray.init(address="auto",namespace="default",ignore_reinit_error=True)   

llm = ByzerLLM(verbose=True)

llm.setup_num_workers(1).setup_gpus_per_worker(0)

chat_name = "baichuan_chat2"
if llm.is_model_exist(chat_name):
    llm.undeploy(udf_name=chat_name)

llm.deploy(model_path="",
           pretrained_model_type="saas/baichuan",
           udf_name=chat_name,
           infer_params={
            "saas.api_key":"xxxxxxx",                        
            "saas.model":"Baichuan2-Turbo"
           })

llm.chat_oai(model=chat_name,conversations=[{
    "role":"user",
    "content":"你好",
}])           
```

针对 `saas.model` 参数，这里有一些枚举值可供选择：

1. Baichuan2-Turbo
2. Baichuan-Text-Embedding

### 通义千问（qianwen）

```python
from byzerllm.utils.client import ByzerLLM
llm = ByzerLLM()

llm.setup_num_workers(1).setup_gpus_per_worker(0)

chat_name = "qianwen_chat"

llm.deploy(model_path="",
           pretrained_model_type="saas/qianwen",
           udf_name=chat_name,
           infer_params={
            "saas.api_key":"xxxxxxx",            
            "saas.model":"qwen-turbo"
           })

## here you can use `stream_chat_oai`
v = llm.stream_chat_oai(model=chat_name,conversations=[{
    "role":"user",
    "content":"你好，你是谁",
}],llm_config={"gen.incremental_output":False})

for t in v:
    print(t,flush=True)           
```

针对 `saas.model` 参数，这里有几个预设的枚举值选项：

1. qwen-turbo
2. qwen-max

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

import ray

from byzerllm.utils.client import ByzerLLM

ray.init(address="auto",namespace="default",ignore_reinit_error=True)  

llm = ByzerLLM()

llm.setup_num_workers(1).setup_gpus_per_worker(0)

chat_name = "openai_chat"

llm.deploy(model_path="",
           pretrained_model_type="saas/official_openai",
           udf_name=chat_name,
           infer_params={
            "saas.api_key":"xxxx",            
            "saas.model":"gpt-3.5-turbo-1106"
           })
```

若您需要用到网络代理，可以尝试运行如下代码来配置代理设置：

```python
llm.deploy(model_path="",
           pretrained_model_type="saas/official_openai",
           udf_name=chat_name,
           infer_params={
            "saas.api_key":"xxxx",            
            "saas.model":"gpt-3.5-turbo-1106"
            "saas.base_url": "http://my.test.server.example.com:8083",
            "saas.proxies":"http://my.test.proxy.example.com"
            "saas.local_address":"0.0.0.0"
           })
```


### 智谱（zhipu）

```python
import ray

from byzerllm.utils.client import ByzerLLM

ray.init(address="auto",namespace="default",ignore_reinit_error=True)  

llm = ByzerLLM()

llm.setup_num_workers(1).setup_gpus_per_worker(0)

chat_name = "zhipu_chat"

llm.deploy(model_path="",
           pretrained_model_type="saas/zhipu",
           udf_name=chat_name,
           infer_params={
            "saas.api_key":"xxxx",            
            "saas.model":"glm-4"
           })
```

针对 `saas.model` 参数，这里有几个预设的枚举值选项：

1. glm-4
2. embedding-2

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

### 星火（sparkdesk）

```python
import ray

from byzerllm.utils.client import ByzerLLM

ray.init(address="auto",namespace="default",ignore_reinit_error=True)

llm = ByzerLLM()

llm.setup_num_workers(1).setup_gpus_per_worker(0)

chat_name = "sparkdesk_saas"

if llm.is_model_exist(chat_name):
  llm.undeploy(udf_name=chat_name)

llm.deploy(model_path="",
           pretrained_model_type="saas/sparkdesk",
           udf_name=chat_name,
           infer_params={
             "saas.appid":"xxxxxxx",
             "saas.api_key":"xxxxxxx",
             "saas.api_secret":"xxxxxxx",
             "saas.gpt_url":"wss://spark-api.xf-yun.com/v3.1/chat",
             "saas.domain":"generalv3"
           })

v = llm.chat_oai(model=chat_name,conversations=[{
  "role":"user",
  "content":"your prompt content",
}])
```

SparkDesk V1.5 版本请求链接，关联的域名参数为“general”：
`wss://spark-api.xf-yun.com/v1.1/chat`  

SparkDesk V2 版本请求链接，关联的域名参数为“generalv2”：
`wss://spark-api.xf-yun.com/v2.1/chat`  

SparkDesk V3 版本请求链接，关联的域名参数更新为“generalv3”（现已支持函数调用功能）：
`wss://spark-api.xf-yun.com/v3.1/chat`  

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

---

## 预训练

在这一部分，我们会讲解如何利用 Byzer-llm 对大型语言模型进行预训练。不过目前来看，Byzer-SQL 中的预训练功能更为成熟，因此我们将聚焦于在 Byzer-SQL 中展示预训练这一功能。

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

因为深速训练保存的模型文件格式与拥抱脸书所使用的模型文件格式互不兼容，我们必须将深速模型文件转换为拥抱脸书能够识别的格式。下面这段代码就是用来实现这一转换任务的。

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

现在，你已经可以顺利部署经过转换的模型，将其投入实际应用了：

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

## 微调

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

你可以在Ray仪表板中查看微调作业进程（finetune actor），其名称通常是 `sft-william-xxxxx`。

待微调作业完成之后，你可以获取到模型路径，这样就可以部署这个经过微调的模型了。

以下是微调作业进程的日志记录：

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

你可以从路径 `/home/byzerllm/projects/sft/jobs/sft-william-20230809-13-04-48-674fd1b9-2fc1-45b9-9d75-7abf07cb84cb/finetune_model/final` 下载已完成微调的模型，或者将模型复制到Ray集群中的所有其他节点上。

现在，尝试部署这个经过微调的模型：

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

Byzer-LLM 利用 QLora 对模型进行微调，你可以通过以下代码将微调后的模型与原始模型进行合并：

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

## 文章

1. [一口气集成那些个大模型你也试试](https://www.51xpage.com/ai/yi-kou-qi-ji-cheng-na-xie-ge-da-mo-xing-ni-ye-shi-shi-unknown-unknown-man-man-xue-ai006/)
2. [Byzer-LLM 快速体验智谱 GLM-4](https://mp.weixin.qq.com/s/Zhzn_C9-dKP4Nq49h8yUxw)
3. [函数实现越通用越好？来看看 Byzer-LLM 的 Function Implementation 带来的编程思想大变化](https://mp.weixin.qq.com/s/_Sx0eC0WqC2M4K1JY9f49Q)
4. [Byzer-LLM 之 QWen-VL-Chat/StableDiffusion多模态读图，生图](https://mp.weixin.qq.com/s/x4g66QvocE5dUlnL1yF9Dw)
5. [基于Byzer-Agent 框架开发智能数据分析工具](https://mp.weixin.qq.com/s/BcoHUEXF24wTjArc7mwNaw)
6. [Byzer-LLM 支持同时开源和SaaS版通义千问](https://mp.weixin.qq.com/s/VvzMUV654D7IO0He47nv3A)
7. [给开源大模型带来Function Calling、 Respond With Class](https://mp.weixin.qq.com/s/GTVCYUhR_atYMX9ymp0eCg)







