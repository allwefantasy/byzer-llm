import asyncio
from typing import List, Dict, Any
from byzerllm.apps.byzer_storage.simple_api import ByzerStorage
import time
import json
import os
import concurrent.futures
import io
import sys
from contextlib import redirect_stdout, redirect_stderr


class MemoryManager:
    _queue = asyncio.Queue()
    _is_processing = False

    def __init__(self, storage: ByzerStorage, base_dir: str):
        self.storage = storage
        home = os.path.expanduser("~")
        self.base_dir = base_dir or os.path.join(home, ".auto-coder")
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

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
    
    async def memorize(self, name: str, memories: List[str]):
        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(
            self.thread_pool, self._memorize_with_logs, name, memories
        )
        print(f"Memorization for {name} completed. Output:")
        print(output)

    def _memorize_with_logs(self, name: str, memories: List[str]) -> str:
        logs_dir = os.path.join(self.base_dir, "storage", "logs", "memorize")
        os.makedirs(logs_dir, exist_ok=True)

        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            self._memorize(name, memories)
        v = output_buffer.getvalue()
        with open(f"{logs_dir}/{name}.log", "w") as f:
            f.write(v)
        return v

    def _memorize(self, name: str, memories: List[str]):
        data = []
        for memory in memories:
            item = {
                "text": memory,
            }
            data.append(item)

        base_model_dir = os.path.join(self.base_dir, "storage", "models")
        llama_model = os.path.join(
            base_model_dir, "meta-llama", "Meta-Llama-3-8B-Instruct-GPTQ"
        )

        loras_dir = os.path.join(self.base_dir, "storage", "loras")
        dataset_dir = os.path.join(self.base_dir, "storage", "datasets", name)

        os.makedirs(loras_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        with open(f"{dataset_dir}/data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        with open(f"{dataset_dir}/dataset_info.json", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"data": {"file_name": "data.json", "columns": {"prompt": "text"}}},
                    indent=2,
                )
            )

        args = dict(
            stage="pt",
            do_train=True,
            model_name_or_path=llama_model,
            dataset="data",
            dataset_dir=dataset_dir,
            cutoff_len=1024,
            max_samples=10,
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

        tuner.run_exp(args)


