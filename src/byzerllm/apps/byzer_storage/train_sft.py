import json
import os
import concurrent.futures
from typing import List, Dict, Any
import byzerllm
from .memory_model_based import read_alpaca_zh
from loguru import logger
from pydantic import BaseModel
from byzerllm.apps.utils import TagExtractor, Tag


class QAPair(BaseModel):
    question: str
    answer: str


@byzerllm.prompt()
def _convert_pretrain_text_to_instruction(text: str, num: int = 10) -> str:
    """
    根据提供的信息，生成多个相关的问题，这些问题的回答，最终要能覆盖里面所有的信息,生成 {{num}} 个。

    下面是一些信息：

    {{text}}

    现在请开始生成问题，每个问题使用<_question_>标签包裹，每个回答使用<_answer_>标签包裹，最后
    每个一组问题和回答使用<_group_>标签包裹。
    """


@byzerllm.prompt()
def get_more(text: str, questions_and_answers: str, num: int = 10) -> str:
    """
    下面是一些问答信息：

    {{text}}

    下面是已有的问题和回答：

    {{questions_and_answers}}

    下面是已经生成的问题和回答，请继续生成{{num}}个问题和回答，不要和前面的重复，请将每个问题使用<_question_>标签包裹，每个回答使用<_answer_>标签包裹，最后
    每个一组问题和回答使用<_group_>标签包裹。
    """


def to_qa_pairs(text: str, llm, num: int = 30) -> List[QAPair]:
    print(f"Generating QA pairs for {text}", flush=True)
    v = _convert_pretrain_text_to_instruction.with_llm(llm).run(text)

    max_turns = int(num / 10)
    if max_turns < 1:
        max_turns = 1

    while max_turns > 0:
        t = get_more.with_llm(llm).run(text, v)
        v += t
        max_turns -= 1

    # format_v = self._format.with_llm(llm).run(v)
    root_tag = TagExtractor(v).extract()
    qa_pairs = []
    # _group_
    for item in root_tag.content:
        qas = item.content
        if len(qas) == 2:
            if (
                qas[0].start_tag == "<_question_>"
                and qas[0].end_tag == "</_question_>"
                and qas[1].start_tag == "<_answer_>"
                and qas[1].end_tag == "</_answer_>"
            ):
                qa_pairs.append(QAPair(question=qas[0].content, answer=qas[1].content))

    print(f"Generated {len(qa_pairs)} QA pairs.", flush=True)
    logger.info(f"Generated {len(qa_pairs)} QA pairs.")
    return qa_pairs


def train_sft(    
    name: str,
    memories: List[str],
    options: Dict[str, Any],
    dataset_dir: str,
    loras_dir: str,
    model_name: str,
):
    data = []
    min_samples = options.pop("min_samples", 1000)

    if memories:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            llm = byzerllm.ByzerLLM().from_default_model(model_name)
            for memory in memories:
                futures.append(executor.submit(to_qa_pairs, memory, llm))

            for future in concurrent.futures.as_completed(futures):
                qa_pairs = future.result()
                for qa_pair in qa_pairs:
                    item = {
                        "instruction": qa_pair.question,
                        "input": "",
                        "output": qa_pair.answer,
                    }
                    data.append(item)
    else:
        v = read_alpaca_zh()
        data.extend(v)

    if len(data) < min_samples:
        data = data * (min_samples // len(data) + 1)

    with open(f"{dataset_dir}/data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(f"{dataset_dir}/dataset_info.json", "w", encoding="utf-8") as f:
        r = {"data": {"file_name": "data.json"}}
        f.write(json.dumps(r, indent=2, ensure_ascii=False))

    args = dict(
        stage="sft",
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
