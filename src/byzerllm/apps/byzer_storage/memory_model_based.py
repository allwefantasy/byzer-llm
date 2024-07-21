import asyncio
from typing import List, Dict, Any

import ray.experimental
from byzerllm.apps.byzer_storage.simple_api import ByzerStorage
import time
import json
import os
import concurrent.futures
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import queue
import threading
import ray
import byzerllm
from typing import List, Union, Dict
from pydantic import BaseModel
from byzerllm.apps.utils import TagExtractor, Tag
import importlib
from loguru import logger


class QAPair(BaseModel):
    question: str
    answer: str


def read_alpaca_zh():
    with importlib.resources.path(
        "byzerllm.apps.byzer_storage", "alpaca_zh.json"
    ) as json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)


class MemoryManager:
    _queue = asyncio.Queue()
    _is_processing = False

    def __init__(
        self,
        storage: ByzerStorage,
        base_dir: str,
        model_name: str = "deepseek_chat",
        remote: bool = True,
    ):
        self.storage = storage
        home = os.path.expanduser("~")
        self.base_dir = base_dir or os.path.join(home, ".auto-coder")
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.log_file = None
        self.remote = remote
        self.model_name = model_name

    @classmethod
    async def add_to_queue(cls, name: str, memories: List[str]):
        await cls._queue.put((name, memories))
        if not cls._is_processing:
            asyncio.create_task(cls.process_queue())

    @classmethod
    async def process_queue(cls):
        cls._is_processing = True
        while not cls._queue.empty():
            name, memories = await cls._queue.get()
            instance = cls.get_instance()
            await instance.memorize(name, memories)
            cls._queue.task_done()
        cls._is_processing = False

    async def memorize(
        self, name: str, memories: List[str], options: Dict[str, Any] = {}
    ):
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.thread_pool, self._memorize_with_logs, name, memories, options
            )
        finally:
            if self.remote:
                ray.actor.exit_actor()

        print(f"Memorization for {name} completed.")

    def _memorize_with_logs(
        self, name: str, memories: List[str], options: Dict[str, Any] = {}
    ):

        if self.remote:
            self._memorize(name, memories, options=options)
            return

        logs_dir = os.path.join(self.base_dir, "storage", "logs", "memorize")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, f"{name}.log")

        log_queue = queue.Queue()
        stop_event = threading.Event()

        # Start the log writer thread
        log_writer_thread = threading.Thread(
            target=self._log_writer, args=(log_queue, log_file, stop_event)
        )
        self.log_file = log_file
        log_writer_thread.start()

        class QueueStream:
            def __init__(self, queue):
                self.queue = queue

            def write(self, msg):
                self.queue.put(msg)

            def flush(self):
                pass

        queue_stream = QueueStream(log_queue)

        with redirect_stdout(queue_stream), redirect_stderr(queue_stream):
            self._memorize(name, memories)

        # Signal the log writer to stop and wait for it to finish
        stop_event.set()
        log_writer_thread.join()

    def _log_writer(
        self, log_queue: queue.Queue, log_file: str, stop_event: threading.Event
    ):
        with open(log_file, "w") as f:
            while not stop_event.is_set() or not log_queue.empty():
                try:
                    msg = log_queue.get(timeout=0.1)
                    f.write(msg)
                    f.flush()
                except queue.Empty:
                    continue

    @byzerllm.prompt()
    def _convert_pretrain_text_to_instruction(self, text: str) -> str:
        """
        根据提供的信息，生成多个相关的问题，这些问题的回答，最终要能覆盖里面所有的信息,生成的问题越多越好。

        下面是一些信息：

        {{text}}

        现在请开始生成问题，每个问题使用<_question_>标签包裹，每个回答使用<_answer_>标签包裹，最后
        每个一组问题和回答使用<_group_>标签包裹。
        """

    @byzerllm.prompt()
    def _format(self, text: str) -> str:
        """
        下面是一些问答信息：

        {{text}}


        请将每个问题使用<_question_>标签包裹，每个回答使用<_answer_>标签包裹，最后
        每个一组问题和回答使用<_group_>标签包裹。
        """

    def to_qa_pairs(self, text: str, llm) -> List[QAPair]:
        print(f"Generating QA pairs for {text}", flush=True)
        v = self._convert_pretrain_text_to_instruction.with_llm(llm).run(text)
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
                    qa_pairs.append(
                        QAPair(question=qas[0].content, answer=qas[1].content)
                    )

        print(f"Generated {len(qa_pairs)} QA pairs.", flush=True)
        logger.info(f"Generated {len(qa_pairs)} QA pairs.")
        return qa_pairs

    def _memorize(
        self,
        name: str,
        memories: List[Union[str, Dict[str, Any]]],
        options: Dict[str, Any] = {},
    ):
        data = []
        stage = options.get("stage", "pt")
        min_samples = options.pop("min_samples", 1000)
        base_model_dir = os.path.join(self.base_dir, "storage", "models")
        llama_model = os.path.join(
            base_model_dir, "meta-llama", "Meta-Llama-3-8B-Instruct-GPTQ"
        )

        loras_dir = os.path.join(self.base_dir, "storage", "loras")
        dataset_dir = os.path.join(self.base_dir, "storage", "datasets", name)

        os.makedirs(loras_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        if stage == "pt":
            for memory in memories:
                item = {
                    "text": memory,
                }
                data.append(item)
        elif stage == "sft":
            if memories:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = []
                    llm = byzerllm.ByzerLLM().from_default_model(self.model_name)
                    for memory in memories:
                        futures.append(executor.submit(self.to_qa_pairs, memory, llm))

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
            if stage == "pt":
                r["data"]["columns"] = {"prompt": "text"}
            f.write(
                json.dumps(
                    r,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        args = dict(
            stage="pt",
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
