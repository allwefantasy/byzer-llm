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


class MemoryManager:
    _queue = asyncio.Queue()
    _is_processing = False

    def __init__(self, storage: ByzerStorage, base_dir: str, remote: bool = True):
        self.storage = storage
        home = os.path.expanduser("~")
        self.base_dir = base_dir or os.path.join(home, ".auto-coder")
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.log_file = None
        self.remote = remote

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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.thread_pool, self._memorize_with_logs, name, memories, options
        )
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

    def _memorize(self, name: str, memories: List[str], options: Dict[str, Any] = {}):
        # target_length = 1024 * 10 * 10
        # original_memories = memories.copy()
        # while sum(len(memory) for memory in memories) < target_length:
        #     memories.extend(original_memories)

        # The rest of the _memorize method remains unchanged
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
            json.dump(data, f, indent=2, ensure_ascii=False)

        with open(f"{dataset_dir}/dataset_info.json", "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"data": {"file_name": "data.json", "columns": {"prompt": "text"}}},
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

        try:
            tuner.run_exp({**args, **options})
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if self.remote:
                ray.actor.exit_actor()
