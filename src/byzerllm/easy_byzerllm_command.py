import argparse
from byzerllm.byzerllm_command import main as byzerllm_main

MODEL_INFER_PARAMS_MAP = {
    "gpt-3.5-turbo-0125": 'saas.api_key=${MODEL_OPENAI_TOKEN}',
    "text-embedding-3-small": 'saas.api_key=${MODEL_OPENAI_TOKEN}',
    "deepseek-chat": 'saas.base_url="https://api.deepseek.com/v1" saas.api_key=${MODEL_DEEPSEEK_TOKEN}',
    "moonshot-v1-32k": 'saas.api_key=${MODEL_KIMI_TOKEN} saas.base_url="https://api.moonshot.cn/v1"',
    "qwen1.5-32b-chat": 'saas.api_key=${MODEL_QIANWEN_TOKEN}',
    "alibaba/Qwen1.5-110B-Chat": 'saas.base_url="https://api.siliconflow.cn/v1" saas.api_key=${MODEL_silcon_flow_TOKEN}',
    "deepseek-ai/deepseek-v2-chat": 'saas.base_url="https://api.siliconflow.cn/v1" saas.api_key=${MODEL_silcon_flow_TOKEN}',
    "alibaba/Qwen2-72B-Instruct": 'saas.base_url="https://api.siliconflow.cn/v1" saas.api_key=${MODEL_silcon_flow_TOKEN}',
    "qwen-vl-chat-v1": 'saas.api_key=${MODEL_QIANWEN_TOKEN}',
    "qwen-vl-max": 'saas.api_key=${MODEL_2_QIANWEN_TOKEN}',
    "yi-vision": 'saas.api_key=${MODEL_YI_TOKEN} saas.base_url=https://api.lingyiwanwu.com/v1'
}

def main():
    parser = argparse.ArgumentParser(description="Easy ByzerLLM command line interface")
    parser.add_argument("command", type=str, help="Command to execute, e.g. 'start'")
    parser.add_argument("model", type=str, help="Model name to deploy")
    parser.add_argument("--token", type=str, required=True, help="The model token")
    parser.add_argument("--ray_address", default="auto", help="Ray cluster address to connect to")
    parser.add_argument("--infer_params", type=str, default="", help="Infer params for the model")
    
    args = parser.parse_args()

    if args.command == "deploy":
        byzerllm_main([
            "deploy",
            "--model", args.model,
            "--ray_address", args.ray_address,
            "--infer_params", args.infer_params or MODEL_INFER_PARAMS_MAP.get(args.model, "")
        ])
    else:
        print(f"Unknown command: {args.command}")