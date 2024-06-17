import argparse
from byzerllm.byzerllm_command import main as byzerllm_main

def main():
    parser = argparse.ArgumentParser(description='Easy ByzerLLM command line interface')
    parser.add_argument('command', type=str, help='Command to execute')
    parser.add_argument('--token', type=str, required=True, help='Token for authentication')
    args = parser.parse_args()

    if args.command == 'start':
        byzerllm_main(['deploy', '--token', args.token])