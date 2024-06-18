# Easy ByzerLLM 使用指南

Easy ByzerLLM 是一个简化版的 ByzerLLM 命令行工具,旨在让用户更方便地部署和使用大语言模型。本文档将介绍如何使用 Easy ByzerLLM 的主要功能。

## 安装

首先,确保你已经安装了 ByzerLLM。如果还没有安装,可以参考主文档中的安装步骤。

## 快速开始

Easy ByzerLLM 提供了三个主要命令:

1. `deploy`: 部署一个模型
2. `undeploy`: 取消部署一个模型 
3. `chat`: 与已部署的模型聊天

下面我们将详细介绍每个命令的使用方法。

### 部署模型

使用 `deploy` 命令可以快速部署一个模型。基本用法如下:

```bash
easy_byzerllm deploy <model_name> --token <your_token>
```

其中,`<model_name>` 是要部署的模型名称,`<your_token>` 是模型的访问令牌。

目前可用的模型以及对应的 base_url 如下:

- `gpt-3.5-turbo-0125`: OpenAI 的 GPT-3.5-turbo 模型,token申请: OpenAI 官网
- `text-embedding-3-small`: OpenAI 的文本嵌入模型,token申请: OpenAI 官网  
- `deepseek-chat`: DeepSeek 的聊天模型,token申请: DeepSeek 官网
- `deepseek-coder`: DeepSeek 的代码生成模型,token申请: DeepSeek 官网
- `moonshot-v1-32k`: Moonshot 的聊天模型,token申请: Moonshot 官网
- `qwen1.5-32b-chat`: 启文 1.5 的 32B 聊天模型,申请： Qwen 官网
- `alibaba/Qwen1.5-110B-Chat`: 启文 1.5 的 110B 聊天模型, Token 申请： 硅基流动官网
- `deepseek-ai/deepseek-v2-chat`: DeepSeek v2 聊天模型,Token 申请： 硅基流动官网
- `alibaba/Qwen2-72B-Instruct`: 启文 2 的 72B 指令微调模型,Token 申请： 硅基流动官网
- `qwen-vl-chat-v1`: 启文的视觉语言聊天模型,Token 申请： Qwen 官网
- `qwen-vl-max`: 启文的 Max 视觉语言聊天模型,Token 申请： Qwen 官网
- `yi-vision`: Yi 的视觉语言模型, Token 申请： Yi（01万物） 官网

例如,要部署 `gpt-3.5-turbo-0125` 模型,可以运行:

```bash  
easy_byzerllm deploy gpt-3.5-turbo-0125 --token your_openai_api_key
```

部署完成后,你就可以开始与模型聊天了。

### 取消部署

如果不再需要使用某个模型,可以使用 `undeploy` 命令将其取消部署,释放资源。用法如下:

```bash
easy_byzerllm undeploy <model_name>
```

其中,`<model_name>` 是要取消部署的模型名称。例如:

```bash  
easy_byzerllm undeploy gpt-3.5-turbo-0125
```

### 聊天

使用 `chat` 命令可以与已部署的模型进行聊天。用法如下:

```bash
easy_byzerllm chat <model_name> <your_query>
```

其中,`<model_name>` 是要聊天的模型名称,`<your_query>` 是你的问题或输入。例如:

```bash
easy_byzerllm chat gpt-3.5-turbo-0125 "Hello, how are you?"  
```

模型会给出相应的回复。

## 高级选项

除了基本的部署和聊天功能,Easy ByzerLLM 还提供了一些高级选项,可以通过命令行参数进行配置。例如:

- `--ray_address`: 指定 Ray 集群的地址
- `--infer_params`: 设置推理参数
- `--force`: 强制取消部署模型

你可以运行 `easy_byzerllm --help` 查看所有可用的选项和说明。

## 总结

Easy ByzerLLM 提供了一种简单快捷的方式来部署和使用大语言模型。通过几个简单的命令,你就可以开始与 AI 助手聊天,提高工作效率。希望这个工具能为你带来便利和帮助!