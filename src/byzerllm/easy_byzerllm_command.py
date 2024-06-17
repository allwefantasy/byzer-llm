import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Easy ByzerLLM command line interface')
    parser.add_argument('command', type=str, help='Command to execute')
    parser.add_argument('--token', type=str, required=True, help='Token for authentication')
    args = parser.parse_args()

    if args.command == 'start':
        subprocess.run(['python', 'byzerllm_command.py', 'deploy', '--token', args.token])