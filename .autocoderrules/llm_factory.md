
---
description: 创建LLM实例的工厂方法
globs: "src/byzerllm/utils/llms.py"
alwaysApply: false
---

# LLM工厂模式

## 简要说明
提供统一接口创建ByzerLLM或SimpleByzerLLM实例，支持pro/lite两种模式及多模型配置。自动处理模型部署和配置，简化LLM实例获取流程。

## 典型用法
```python
from byzerllm.utils.llms import get_single_llm
from byzerllm import ByzerLLM, SimpleByzerLLM

# 获取pro模式的LLM实例 (需要本地部署)
pro_llm: ByzerLLM = get_single_llm(
    model_names="qwen",  # 模型名称或逗号分隔的多个名称
    product_mode="pro"   # 产品模式: "pro"或"lite"
)

# 获取lite模式的LLM实例 (SaaS服务)
lite_llm: SimpleByzerLLM = get_single_llm(
    model_names="openai/gpt-4",  # SaaS模型名称
    product_mode="lite"          # 使用SaaS服务
)

# 使用LLM
response = lite_llm.chat_oai([{"role":"user","content":"Hello"}])
```

## 依赖说明
- byzerllm: 核心LLM库
- autocoder.models: 用于获取模型配置信息 (lite模式)
- 环境要求:
  - pro模式需要本地模型部署
  - lite模式需要API key等配置

## 学习来源
从src/byzerllm/utils/llms.py中的get_single_llm函数提取
