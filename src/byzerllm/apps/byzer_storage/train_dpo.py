import json
import os
from typing import List, Dict, Any
import byzerllm


@byzerllm.prompt()
def generate_dpo_data(memory: str) -> str:
    """
    根据以下内容，生成一个DPO（Direct Preference Optimization）训练数据样本。
    每个样本应包含一个提示（prompt），一个首选回答（preferred_output）和一个拒绝回答（rejected_output）。
    首选回答应该是高质量、有帮助的回答，而拒绝回答应该是质量较差或不太有帮助的回答。

    内容：
    {{memory}}

    请生成DPO训练数据，并以JSON格式返回，格式如下：
    {
        "prompt": "用户提问或指令",
        "preferred_output": "高质量、有帮助的回答",
        "rejected_output": "质量较差或不太有帮助的回答"
    }
    """


def train_dpo(
    self,
    name: str,
    memories: List[str],
    options: Dict[str, Any],
    dataset_dir: str,
    loras_dir: str,
    llama_model: str,
):
    data = []
    min_samples = options.pop("min_samples", 1000)

    llm = byzerllm.ByzerLLM().from_default_model(self.model_name)
    for memory in memories:
        dpo_sample = generate_dpo_data.with_llm(llm).run(memory)
        data.append(json.loads(dpo_sample))

    if len(data) < min_samples:
        data = data * (min_samples // len(data) + 1)

    with open(f"{dataset_dir}/data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(f"{dataset_dir}/dataset_info.json", "w", encoding="utf-8") as f:
        r = {"data": {"file_name": "data.json"}}
        f.write(json.dumps(r, indent=2, ensure_ascii=False))

    args = dict(
        stage="dpo",
        do_train=True,
        model_name_or_path=llama_model,
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
