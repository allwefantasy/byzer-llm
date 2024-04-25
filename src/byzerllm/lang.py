import os

# 命令和参数的中英文映射字典
locales = {
    "desc": {
        "en": "Byzer-LLM command line tool",
        "zh": "Byzer-LLM 命令行工具"
    },
    "help_deploy": {
        "en": "Deploy a model",
        "zh": "部署一个模型"
    },
    "help_ray_address": {
        "en": "Ray cluster address",
        "zh": "Ray 集群地址"
    },
    "help_num_workers": {
        "en": "Number of model workers",
        "zh": "模型工作节点数"
    },
    "help_gpus_per_worker": {
        "en": "Number of GPUs per worker",
        "zh": "每个工作节点的 GPU 数"
    },
     "help_cpus_per_worker": {
        "en": "Number of CPUs per worker",
        "zh": "每个工作节点的 CPU 数"
    },
    "help_model_path": {
        "en": "Local model directory path",
        "zh": "本地模型目录路径"
    },
    "help_pretrained_model_type": {
        "en": "Pretrained model type",
        "zh": "预训练模型类型"
    },
    "help_udf_name": {
        "en": "Deployed model name",
        "zh": "部署后的模型名称"
    },
    "help_infer_params": {
        "en": "Model inference parameters",
        "zh": "模型推理参数"
    },
    "help_infer_backend": {
        "en": "Model inferrence Backend",
        "zh": "模型推理后端"
    },
    "help_query": {
        "en": "Query a deployed model",
        "zh": "查询一个已部署的模型"
    },
    "help_query_model": {
        "en": "Deployed model UDF name",
        "zh": "已部署的模型 UDF 名称"
    },
    "help_query_text": {
        "en": "User query/prompt",
        "zh": "用户查询/提示"
    },
    "help_template": {
        "en": "Chat template",
        "zh": "对话模板"
    },
    "deploy_success": {
        "en": "Model {0} deployed successfully",
        "zh": "模型 {0} 部署成功"
    },
    "undeploy_success": {
        "en": "Model {0} undeployed successfully",
        "zh": "模型 {0} 卸载成功"
    },
    "already_deployed": {
        "en": "Model {0} already deployed",
        "zh": "模型 {0} 已经部署过了"
    }
}

# 获取系统语言环境
lang = os.getenv("LANG", "en").split(".")[0]
if lang.startswith("zh"):
    lang = "zh"
else:
    lang = "en"