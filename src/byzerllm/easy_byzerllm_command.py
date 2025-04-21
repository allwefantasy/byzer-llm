
import argparse
import re
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
    # Placeholder, assuming model name can directly be used as instance name
    # Or implement specific mapping logic if needed
    return model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Easy ByzerLLM command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Deploy command parser
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a model")
    deploy_parser.add_argument("model", type=str, help="Model name to deploy (e.g., gpt-3.5-turbo-0125)")
    deploy_parser.add_argument(
        "--token", type=str, required=True, help="The API token for the model service"
    )
    deploy_parser.add_argument(
        "--ray_address", default="auto", help="Ray cluster address (default: auto)"
    )
    deploy_parser.add_argument(
        "--infer_params", type=str, default="", help="Custom infer params for the model"
    )
    deploy_parser.add_argument(
        "--base_url", type=str, default="", help="Override base URL for the model service"
    )
    deploy_parser.add_argument(
        "--alias", type=str, default="", help="Alias name for the deployed model instance"
    )

    # Undeploy command parser
    undeploy_parser = subparsers.add_parser("undeploy", help="Undeploy a model")
    undeploy_parser.add_argument("model", type=str, help="Model name or alias to undeploy")
    undeploy_parser.add_argument(
        "--ray_address", default="auto", help="Ray cluster address (default: auto)"
    )
    undeploy_parser.add_argument(
        "--force", action="store_true", help="Force undeployment"
    )

    # Chat command parser
    chat_parser = subparsers.add_parser("chat", help="Chat with a deployed model")
    chat_parser.add_argument("model", type=str, help="Model name or alias to chat with")
    chat_parser.add_argument("query", type=str, help="The user's query or prompt")
    chat_parser.add_argument(
        "--ray_address", default="auto", help="Ray cluster address (default: auto)"
    )

    return parser.parse_args()


def handle_deploy(args):
    instance_name = args.alias or model_to_instance(args.model)
    infer_params = args.infer_params or MODEL_INFER_PARAMS_MAP.get(args.model, "")

    if not infer_params and args.model not in MODEL_INFER_PARAMS_MAP:
         print(f"Warning: No default infer_params found for model '{args.model}'. Deployment might require manual --infer_params.")

    if args.base_url:
        if "saas.base_url" in infer_params:
            # Replace existing base_url
            infer_params = re.sub(r'saas\.base_url=(?:"[^"]*"|[^ ]+)', f'saas.base_url="{args.base_url}"', infer_params)
        else:
            # Add new base_url
            infer_params += f' saas.base_url="{args.base_url}"'

    pretrained_model_type = MODEL_PRETRAINED_TYPE_MAP.get(args.model)
    if not pretrained_model_type:
        # Attempt to infer from common patterns or raise error
        if "qianwen" in args.model or "qwen" in args.model:
             if "vl" in args.model:
                 pretrained_model_type = "saas/qianwen_vl"
             else:
                 pretrained_model_type = "saas/qianwen"
        elif "openai" in args.model or "gpt" in args.model:
             pretrained_model_type = "saas/openai"
        elif "deepseek" in args.model:
             pretrained_model_type = "saas/openai" # Assuming OpenAI compatible API
        elif "moonshot" in args.model:
             pretrained_model_type = "saas/official_openai"
        elif "yi-" in args.model:
             pretrained_model_type = "saas/openai" # Assuming OpenAI compatible API for Yi
        elif "claude" in args.model or "sonnet" in args.model:
             pretrained_model_type = "saas/claude"
        else:
            raise ValueError(
                f"Cannot determine pretrained_model_type for model '{args.model}'. "
                f"Please check MODEL_PRETRAINED_TYPE_MAP or provide it implicitly if needed."
            )
        print(f"Inferred pretrained_model_type: {pretrained_model_type} for model {args.model}")


    # Substitute token securely
    if "${MODEL_TOKEN}" in infer_params:
        infer_params = infer_params.replace("${MODEL_TOKEN}", args.token)
    elif "saas.api_key=" not in infer_params:
         # Add api_key if not present and token is provided
         infer_params += f" saas.api_key={args.token}"
    else:
        # This case might indicate manual infer_params without the placeholder
        print("Warning: MODEL_TOKEN placeholder not found in infer_params. Ensure API key is correctly set if required.")


    byzerllm_args = [
        "deploy",
        "--model", instance_name,
        "--ray_address", args.ray_address,
        "--infer_params", infer_params,
        "--pretrained_model_type", pretrained_model_type,
        # Default resource settings, consider making these configurable if needed
        "--cpus_per_worker", "0.001",
        "--gpus_per_worker", "0",
        "--num_workers", "1",
        "--worker_concurrency", "1000",
    ]
    print(f"Executing ByzerLLM deploy command: {' '.join(byzerllm_args)}")
    byzerllm_main(byzerllm_args)


def handle_undeploy(args):
    instance_name = model_to_instance(args.model) # Use alias if provided during deploy? Needs consistent handling.
    byzerllm_args = [
        "undeploy",
        "--model", instance_name,
        "--ray_address", args.ray_address,
    ]
    if args.force:
        byzerllm_args.append("--force")

    print(f"Executing ByzerLLM undeploy command: {' '.join(byzerllm_args)}")
    byzerllm_main(byzerllm_args)


def handle_chat(args):
    instance_name = model_to_instance(args.model) # Use alias if provided during deploy? Needs consistent handling.
    byzerllm_args = [
        "query", # Note: byzerllm_command uses 'query' not 'chat'
        "--model", instance_name,
        "--ray_address", args.ray_address,
        "--query", args.query,
    ]
    print(f"Executing ByzerLLM query command: {' '.join(byzerllm_args)}")
    byzerllm_main(byzerllm_args)


def main():
    args = parse_args()

    command_handlers = {
        "deploy": handle_deploy,
        "undeploy": handle_undeploy,
        "chat": handle_chat,
    }

    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
    else:
        # This should not happen if subparsers are required=True
        print(f"Error: Unknown command '{args.command}'")
        exit(1)

if __name__ == "__main__":
    main()
