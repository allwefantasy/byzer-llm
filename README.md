# Byzer-LLM

Byzer-LLM is a LLM full lifecycle solution that includes pretrain, fintune, deployment and serving based on Ray.

The key differences between Byzer-LLM and other LLM solutions have two.
The first one is that Byzer-LLM supports Byzer-SQL which is a SQL dialect that can be used to manage the LLM lifecycle as the other solutions only support Python API.

1. Python (alpha)
2. [Byzer-SQL](https://github.com/byzer-org/byzer-lang) (stable)
3. Rest API (todo...)

The second one is that Byzer-LLM is totally based on Ray. This means you can deploy multiple LLM models on a single machine or a cluster. This is very useful for large scale LLM deployment. And Byzer-LLM also supports vLLM/DeepSpeed/Transformers as the inference backend transparently.

## Installation

```bash
pip install -r requirements.txt
pip install -U byzerllm
ray start --head
```

## Usage (Python)

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

The above code will deploy a llama2 model and then use the model to infer the input text. The Python API is very simple and easy to use and it is very useful to explore the LLM model.

## Usage (Byzer-SQL)

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

## Cooperate with Byzer-Retrieval

Byzer-LLM can cooperate with [Byzer-Retrieval](https://github.com/allwefantasy/byzer-retrieval) to build a RAG application. The following code shows how to use Byzer-LLM and Byzer-Retrieval together.

The first step is connect to Ray cluster:

```python
code_search_path=["/home/byzerllm/softwares/byzer-retrieval-lib/"]
env_vars = {"JAVA_HOME": "/home/byzerllm/softwares/jdk-21",
            "PATH":"/home/byzerllm/softwares/jdk-21/bin:/home/byzerllm/.rvm/gems/ruby-3.2.2/bin:/home/byzerllm/.rvm/gems/ruby-3.2.2@global/bin:/home/byzerllm/.rvm/rubies/ruby-3.2.2/bin:/home/byzerllm/.rbenv/shims:/home/byzerllm/.rbenv/bin:/home/byzerllm/softwares/byzer-lang-all-in-one-linux-amd64-3.3.0-2.3.7/jdk8/bin:/usr/local/cuda/bin:/usr/local/cuda/bin:/home/byzerllm/.rbenv/shims:/home/byzerllm/.rbenv/bin:/home/byzerllm/miniconda3/envs/byzerllm-dev/bin:/home/byzerllm/miniconda3/condabin:/home/byzerllm/.local/bin:/home/byzerllm/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/byzerllm/.rvm/bin:/home/byzerllm/.rvm/bin"}

import ray

ray.init(address="auto",namespace="default",
                 job_config=ray.job_config.JobConfig(code_search_path=code_search_path,
                                                      runtime_env={"env_vars": env_vars})
                 )            
```

The second step is to create ByzerLLM/ByzerRetrieval:

```python
from byzerllm.utils.retrieval import ByzerRetrieval
from byzerllm.utils.client import ByzerLLM,LLMRequest,LLMResponse,LLMHistoryItem,LLMRequestExtra
from byzerllm.records import SearchQuery

retrieval = ByzerRetrieval()
retrieval.launch_gateway()

llm = ByzerLLM()
```


Now you can use `llm.chat` and `retrieval.search` to build a RAG application.


## Versions

- 0.1.12: Support Python API (alpha)
- 0.1.5: Support python wrapper for [byzer-retrieval](https://github.com/allwefantasy/byzer-retrieval)


