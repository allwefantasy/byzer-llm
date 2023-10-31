# Byzer-LLM

Byzer-LLM is a LLM full lifecycle solution that includes pretrain, fintune, deployment and serving based on Ray.

The key difference between Byzer-LLM and other LLM solutions is that Byzer-LLM supports using
the following languages to manage the LLM lifecycle:

1. Python (alpha)
2. [Byzer-SQL](https://github.com/byzer-org/byzer-lang) (stable)
3. Rest API (todo...)

## Installation

```bash
pip install -r requirements.txt
pip install -U byzerllm
```

## Usage (Python)

```python
import ray
from byzerllm.utils.client import ByzerLLM,LLMRequest
ray.init(address="auto",namespace="default",ignore_reinit_error=True)

llm = ByzerLLM()
llm.deploy(model_path="/home/byzerllm/models/m3e-base",
           pretrained_model_type="custom/m3e",
           udf_name="emb",infer_params={})

llm.emb("emb",LLMRequest(instruction="hello world"))[0].output
```

## Versions

- 0.1.5: Support python wrapper for [byzer-retrieval](https://github.com/allwefantasy/byzer-retrieval)


