include Makefile-flash-att
include Makefile-vllm

install-torch:
	# Install specific version of torch
	pip install torch --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

run-dev:
	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve bigscience/bloom-560m --sharded

export-requirements:
	poetry export -o requirements.txt -E bnb --without-hashes
