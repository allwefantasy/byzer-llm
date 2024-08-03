import json
import os
from typing import List, Dict, Any


def train_pt(
    name: str,
    memories: List[str],
    options: Dict[str, Any],
    dataset_dir: str,
    loras_dir: str,
    target_model_path: str,
    data_model_name: str,
    model_name: str,
):
    data = []
    min_samples = options.pop("min_samples", 1000)

    for memory in memories:
        item = {
            "text": memory,
        }
        data.append(item)

    if len(data) < min_samples:
        data = data * (min_samples // len(data) + 1)

    with open(f"{dataset_dir}/data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(f"{dataset_dir}/dataset_info.json", "w", encoding="utf-8") as f:
        r = {"data": {"file_name": "data.json", "columns": {"prompt": "text"}}}
        f.write(json.dumps(r, indent=2, ensure_ascii=False))

    args = dict(
        stage="pt",
        do_train=True,
        model_name_or_path=target_model_path,
        dataset="data",
        dataset_dir=dataset_dir,
        cutoff_len=1024,
        max_samples=1000000,
        overwrite_cache=True,
        preprocessing_num_workers=1,
        template="llama3",
        finetuning_type="lora",
        lora_target="all",
        output_dir=f"{loras_dir}/{name}",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        warmup_ratio=0.1,
        save_steps=1000,
        plot_loss=True,
        learning_rate=5e-5,
        num_train_epochs=1000.0,
        max_grad_norm=1.0,
        quantization_bit=4,
        loraplus_lr_ratio=16.0,
        fp16=True,
        ddp_timeout=180000000,
    )
    os.environ["WANDB_DISABLED"] = "true"
    from llamafactory.train import tuner

    tuner.run_exp({**args, **options})
