import torch
import torch.distributed

from pathlib import Path
from typing import Optional, Type
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase
from huggingface_hub import hf_hub_download
import json

from . import CausalLM
from .causal_lm import CausalLMBatch
from .custom_modeling.mpt_modeling import (
    MPTForCausalLM,
)
from ..utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)



class MPTCausalLMBatch(CausalLMBatch):
    pass


class MPTSharded(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16
        else:
            raise NotImplementedError("MPTSharded is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token

        # If model_id is a local path, load the file directly
        local_path = Path(model_id, "config.json")
        if local_path.exists():
            filename = str(local_path.resolve())
        else:
            filename = hf_hub_download(
                model_id, revision=revision, filename="config.json"
            )
        with open(filename, "r") as f:
            config = json.load(f)
        config = PretrainedConfig(**config)
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)

        config.quantize = quantize
        model = MPTForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return MPTCausalLMBatch
