import argparse
from byzerllm.byzerllm_command import main as byzerllm_main

MODEL_INFER_PARAMS_MAP = {
    "gpt-3.5-turbo-0125": "saas.api_key=${MODEL_TOKEN} saas.model=gpt-3.5-turbo-0125",
    "text-embedding-3-small": "saas.api_key=${MODEL_TOKEN} saas.model=text-embedding-3-small",
    "deepseek-chat": 'saas.base_url="https://api.deepseek.com/v1" saas.api_key=${MODEL_TOKEN} saas.model=deepseek-chat',
    "deepseek-coder": 'saas.base_url="https://api.deepseek.com/v1" saas.api_key=${MODEL_TOKEN} saas.model=deepseek-coder',
    "moonshot-v1-32k": 'saas.api_key=${MODEL_TOKEN} saas.base_url="https://api.moonshot.cn/v1" saas.model=moonshot-v1-32k',
    "qwen1.5-32b-chat": "saas.api_key=${MODEL_TOKEN} saas.model=qwen1.5-32b-chat",
    "qwen-max": "saas.api_key=${MODEL_TOKEN} saas.model=qwen-max",
    "alibaba/Qwen1.5-110B-Chat": 'saas.base_url="https://api.siliconflow.cn/v1" saas.api_key=${MODEL_TOKEN} saas.model=alibaba/Qwen1.5-110B-Chat',
    "deepseek-ai/deepseek-v2-chat": 'saas.base_url="https://api.siliconflow.cn/v1" saas.api_key=${MODEL_TOKEN} saas.model=deepseek-ai/deepseek-v2-chat',
    "alibaba/Qwen2-72B-Instruct": 'saas.base_url="https://api.siliconflow.cn/v1" saas.api_key=${MODEL_TOKEN} saas.model=alibaba/Qwen2-72B-Instruct',
    "qwen-vl-chat-v1": "saas.api_key=${MODEL_TOKEN} saas.model=qwen-vl-chat-v1",
    "qwen-vl-max": "saas.api_key=${MODEL_TOKEN} saas.model=qwen-vl-max",
    "yi-vision": "saas.api_key=${MODEL_TOKEN} saas.base_url=https://api.lingyiwanwu.com/v1 saas.model=yi-vision",
    "gpt4o": "saas.api_key=${MODEL_TOKEN} saas.model=gpt-4o",
    "sonnet3.5": "saas.api_key=${MODEL_TOKEN} saas.model=claude-3-5-sonnet-20240620",
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
    "gpt4o": "saas/openai",
    "sonnet3.5": "saas/claude",
}


def model_to_instance(model_name):
    return model_name


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
    deploy_parser.add_argument(
        "--base_url", type=str, default="", help="base url"
    )
    deploy_parser.add_argument(
        "--alias", type=str, default="", help="Alias name for the deployed model"
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
        instance_name = args.alias or model_to_instance(args.model)
        infer_params = args.infer_params or MODEL_INFER_PARAMS_MAP.get(args.model, "")
        
        if args.base_url:
            if "saas.base_url" in infer_params:
                infer_params = re.sub(r"saas.base_url=[^ ]+", f"saas.base_url={args.base_url}", infer_params)
            else:
                infer_params += f" saas.base_url={args.base_url}"

        pretrained_model_type = MODEL_PRETRAINED_TYPE_MAP.get(args.model, "")
        if not pretrained_model_type:
            raise ValueError(f"Pretrained model type not found for model {args.model}")

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
                "1000",
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
