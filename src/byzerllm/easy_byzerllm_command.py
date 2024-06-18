import argparse
from byzerllm.byzerllm_command import main as byzerllm_main

MODEL_INFER_PARAMS_MAP = {
    "gpt-3.5-turbo-0125": "saas.api_key=${MODEL_TOKEN} saas.model=gpt-3.5-turbo-0125",
    "text-embedding-3-small": "saas.api_key=${MODEL_TOKEN}",
    "deepseek-chat": 'saas.base_url="https://api.deepseek.com/v1" saas.api_key=${MODEL_TOKEN}',
    "deepseek-coder": 'saas.base_url="https://api.deepseek.com/v1" saas.api_key=${MODEL_TOKEN}',
    "moonshot-v1-32k": 'saas.api_key=${MODEL_TOKEN} saas.base_url="https://api.moonshot.cn/v1"',
    "qwen1.5-32b-chat": "saas.api_key=${MODEL_TOKEN}",
    "qwen-max": "saas.api_key=${MODEL_TOKEN}",
    "alibaba/Qwen1.5-110B-Chat": 'saas.base_url="https://api.siliconflow.cn/v1" saas.api_key=${MODEL_TOKEN}',
    "deepseek-ai/deepseek-v2-chat": 'saas.base_url="https://api.siliconflow.cn/v1" saas.api_key=${MODEL_TOKEN}',
    "alibaba/Qwen2-72B-Instruct": 'saas.base_url="https://api.siliconflow.cn/v1" saas.api_key=${MODEL_TOKEN}',
    "qwen-vl-chat-v1": "saas.api_key=${MODEL_TOKEN}",
    "qwen-vl-max": "saas.api_key=${MODEL_TOKEN}",
    "yi-vision": "saas.api_key=${MODEL_TOKEN} saas.base_url=https://api.lingyiwanwu.com/v1",
}

import re

MODEL_PRETRAINED_TYPE_MAP = {
    "gpt-3.5-turbo-0125": "saas/openai",
    "text-embedding-3-small": "saas/openai",
    "deepseek-chat": "saas/openai",
    "deepseek-coder": "saas/openai",
    "moonshot-v1-32k": "saas/official_openai",
    "qwen1.5-32b-chat": "saas/qianwen",
    "qwen-max": "saas/qianwen",
    "alibaba/Qwen1.5-110B-Chat": "saas/openai",
    "deepseek-ai/deepseek-v2-chat": "saas/openai",
    "alibaba/Qwen2-72B-Instruct": "saas/openai",
    "qwen-vl-chat-v1": "saas/qianwen_vl",
    "qwen-vl-max": "saas/qianwen_vl",
    "yi-vision": "saas/openai",
}


def model_to_instance(model_name):
    return re.sub(r"[/-]", "_", model_name)


def main():
    parser = argparse.ArgumentParser(description="Easy ByzerLLM command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_parser = subparsers.add_parser("deploy", help="Deploy a model")
    deploy_parser.add_argument("model", type=str, help="Model name to deploy")
    deploy_parser.add_argument(
        "--token", type=str, required=True, help="The model token"
    )
    deploy_parser.add_argument(
        "--ray_address", default="auto", help="Ray cluster address to connect to"
    )
    deploy_parser.add_argument(
        "--infer_params", type=str, default="", help="Infer params for the model"
    )

    undeploy_parser = subparsers.add_parser("undeploy", help="Undeploy a model")
    undeploy_parser.add_argument("model", type=str, help="Model name to undeploy")
    undeploy_parser.add_argument(
        "--ray_address", default="auto", help="Ray cluster address to connect to"
    )
    undeploy_parser.add_argument(
        "--force", action="store_true", help="Force undeploy the model"
    )

    chat_parser = subparsers.add_parser("chat", help="Chat with a deployed model")
    chat_parser.add_argument("model", type=str, help="Model name to chat with")
    chat_parser.add_argument("query", type=str, help="User query")
    chat_parser.add_argument(
        "--ray_address", default="auto", help="Ray cluster address to connect to"
    )

    args = parser.parse_args()
    instance_name = model_to_instance(args.model)

    if args.command == "deploy":
        infer_params = args.infer_params or MODEL_INFER_PARAMS_MAP.get(args.model, "")
        pretrained_model_type = MODEL_PRETRAINED_TYPE_MAP.get(args.model, "")

        infer_params = infer_params.replace("${MODEL_TOKEN}", args.token)

        byzerllm_main(
            [
                "deploy",
                "--model",
                instance_name,
                "--ray_address",
                args.ray_address,
                "--infer_params",
                infer_params,
                "--pretrained_model_type",
                pretrained_model_type,
                "--cpus_per_worker",
                "0.001",
                "--gpus_per_worker",
                "0",
                "--num_workers",
                "1",
                "--worker_concurrency",
                "10",
            ]
        )
    elif args.command == "undeploy":
        target_args = [
            "undeploy",
            "--model",
            instance_name,
            "--ray_address",
            args.ray_address,
        ]
        if args.force:
            target_args.append("--force")
        byzerllm_main(target_args)

    elif args.command == "chat":
        byzerllm_main(
            [
                "query",
                "--model",
                instance_name,
                "--ray_address",
                args.ray_address,
                "--query",
                args.query,
            ]
        )
