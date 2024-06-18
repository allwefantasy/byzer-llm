import argparse
from byzerllm.byzerllm_command import main as byzerllm_main


def main():
    parser = argparse.ArgumentParser(description="Easy ByzerLLM command line interface")
    parser.add_argument("command", type=str, help="Command to execute, e.g. 'start'")
    parser.add_argument("model", type=str, help="Model name to deploy")
    parser.add_argument("--token", type=str, required=True, help="Token for authentication")
    parser.add_argument("--ray_address", default="auto", help="Ray cluster address to connect to")
    
    args = parser.parse_args()

    if args.command == "start":
        byzerllm_main([
            "deploy",
            "--model", args.model, 
            "--token", args.token,
            "--ray_address", args.ray_address
        ])
    else:
        print(f"Unknown command: {args.command}")
