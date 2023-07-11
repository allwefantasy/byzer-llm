import argparse
import json
import os
import shutil
from collections import defaultdict
from inspect import signature
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Set, Tuple

import torch

from huggingface_hub import CommitInfo, CommitOperationAdd, Discussion, HfApi, hf_hub_download
from huggingface_hub.file_download import repo_folder_name
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.pipelines.base import infer_framework_load_model


COMMIT_DESCRIPTION = """
This is an automated PR created with https://huggingface.co/spaces/safetensors/convert
This new file is equivalent to `pytorch_model.bin` but safe in the sense that
no arbitrary code can be put into it.
These files also happen to load much faster than their pytorch counterpart:
https://colab.research.google.com/github/huggingface/notebooks/blob/main/safetensors_doc/en/speed.ipynb
The widgets on your model page will run using this model even if this is not merged
making sure the file actually works.
If you find any issues: please report here: https://huggingface.co/spaces/safetensors/convert/discussions
Feel free to ignore this PR.
"""

ConversionResult = Tuple[List["CommitOperationAdd"], List[Tuple[str, "Exception"]]]


class AlreadyExists(Exception):
    pass


def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )


def rename(pt_filename: str) -> str:
    filename, ext = os.path.splitext(pt_filename)
    local = f"{filename}.safetensors"
    local = local.replace("pytorch_model", "model")
    return local


def convert_multi(model_dir: str) -> ConversionResult:
    filename = os.path.join(model_dir,"pytorch_model.bin.index.json")
    with open(filename, "r") as f:
        data = json.load(f)

    filenames = set(data["weight_map"].values())
    local_filenames = []
    for filename in filenames:
        pt_filename = os.path.join(model_dir, filename)

        sf_filename = rename(pt_filename)        
        
        convert_file(pt_filename, sf_filename)
        local_filenames.append(sf_filename)

    index = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index, "w") as f:
        newdata = {k: v for k, v in data.items()}
        newmap = {k: rename(v) for k, v in data["weight_map"].items()}
        newdata["weight_map"] = newmap
        json.dump(newdata, f, indent=4)
    local_filenames.append(index)

    operations = [
        CommitOperationAdd(path_in_repo=local.split("/")[-1], path_or_fileobj=local) for local in local_filenames
    ]
    errors: List[Tuple[str, "Exception"]] = []

    return operations, errors


def convert_single(model_dir: str) -> ConversionResult:
    pt_filename = os.path.join(model_dir,"pytorch_model.bin")

    sf_name = "model.safetensors"
    sf_filename = os.path.join(model_dir, sf_name)
    convert_file(pt_filename, sf_filename)
    operations = [CommitOperationAdd(path_in_repo=sf_name, path_or_fileobj=sf_filename)]
    errors: List[Tuple[str, "Exception"]] = []
    return operations, errors


def convert_file(
    pt_filename: str,
    sf_filename: str,
):
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    print(f"Saving to {sf_filename}")
    save_file(loaded, sf_filename, metadata={"format": "pt"})
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def create_diff(pt_infos: Dict[str, List[str]], sf_infos: Dict[str, List[str]]) -> str:
    errors = []
    for key in ["missing_keys", "mismatched_keys", "unexpected_keys"]:
        pt_set = set(pt_infos[key])
        sf_set = set(sf_infos[key])

        pt_only = pt_set - sf_set
        sf_only = sf_set - pt_set

        if pt_only:
            errors.append(f"{key} : PT warnings contain {pt_only} which are not present in SF warnings")
        if sf_only:
            errors.append(f"{key} : SF warnings contain {sf_only} which are not present in PT warnings")
    return "\n".join(errors)


def check_final_model(model_dir: str, folder: str):
    config = os.path.join(model_dir,"config.json")
    # shutil.copy(config, os.path.join(folder, "config.json"))
    config = AutoConfig.from_pretrained(folder)

    _, (pt_model, pt_infos) = infer_framework_load_model(model_dir, config, output_loading_info=True)
    _, (sf_model, sf_infos) = infer_framework_load_model(folder, config, output_loading_info=True)

    if pt_infos != sf_infos:
        error_string = create_diff(pt_infos, sf_infos)
        raise ValueError(f"Different infos when reloading the model: {error_string}")

    pt_params = pt_model.state_dict()
    sf_params = sf_model.state_dict()

    pt_shared = shared_pointers(pt_params)
    sf_shared = shared_pointers(sf_params)
    if pt_shared != sf_shared:
        raise RuntimeError("The reconstructed model is wrong, shared tensors are different {shared_pt} != {shared_tf}")

    sig = signature(pt_model.forward)
    input_ids = torch.arange(10).unsqueeze(0)
    pixel_values = torch.randn(1, 3, 224, 224)
    input_values = torch.arange(1000).float().unsqueeze(0)
    kwargs = {}
    if "input_ids" in sig.parameters:
        kwargs["input_ids"] = input_ids
    if "decoder_input_ids" in sig.parameters:
        kwargs["decoder_input_ids"] = input_ids
    if "pixel_values" in sig.parameters:
        kwargs["pixel_values"] = pixel_values
    if "input_values" in sig.parameters:
        kwargs["input_values"] = input_values
    if "bbox" in sig.parameters:
        kwargs["bbox"] = torch.zeros((1, 10, 4)).long()
    if "image" in sig.parameters:
        kwargs["image"] = pixel_values

    if torch.cuda.is_available():
        pt_model = pt_model.cuda()
        sf_model = sf_model.cuda()
        kwargs = {k: v.cuda() for k, v in kwargs.items()}

    pt_logits = pt_model(**kwargs)[0]
    sf_logits = sf_model(**kwargs)[0]

    torch.testing.assert_close(sf_logits, pt_logits)
    print(f"Model {model_dir} is ok !")


def convert_generic(model_dir: str, filenames: Set[str]) -> ConversionResult:
    operations = []
    errors = []

    extensions = set([".bin", ".ckpt"])
    for filename in filenames:
        prefix, ext = os.path.splitext(filename)
        if ext in extensions:
            pt_filename = os.path.join(model_dir,filename) 
            dirname, raw_filename = os.path.split(filename)
            if raw_filename == "pytorch_model.bin":
                # XXX: This is a special case to handle `transformers` and the
                # `transformers` part of the model which is actually loaded by `transformers`.
                sf_in_repo = os.path.join(dirname, "model.safetensors")
            else:
                sf_in_repo = f"{prefix}.safetensors"
            sf_filename = os.path.join(model_dir, sf_in_repo)
            try:
                convert_file(pt_filename, sf_filename)
                operations.append(CommitOperationAdd(path_in_repo=sf_in_repo, path_or_fileobj=sf_filename))
            except Exception as e:
                errors.append((pt_filename, e))
    return operations, errors


def convert( model_dir: str,library_name: str = "transformers") -> Tuple["CommitInfo", List["Exception"]]:
    filenames = set(os.listdir(model_dir))    

    if library_name == "transformers":
        if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
            operations, errors = convert_single(model_dir)
        elif os.path.exists(os.path.join(model_dir, "pytorch_model.bin.index.json")):
            operations, errors = convert_multi(model_dir)
        else:
            raise RuntimeError(f"Model {model_dir} doesn't seem to be a valid pytorch model. Cannot convert")
        check_final_model(model_dir, model_dir)
    else:
        operations, errors = convert_generic(model_dir, filenames)            
    


if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert automatically some weights of LLM to `safetensors` format.
    It is PyTorch exclusive for now.
    It works by downloading the weights (PT), converting them locally.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "model_dir",
        type=str,
        help="The model path to convert",
    )    
  
    args = parser.parse_args()
    model_dir = args.model_dir
    print(model_dir)
    convert(model_dir)