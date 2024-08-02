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
import importlib
from loguru import logger

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
            from .train_pt import train_pt
            train_pt(self, name, memories, options, dataset_dir, loras_dir, llama_model)
        elif stage == "sft":
            from .train_sft import train_sft
            train_sft(self, name, memories, options, dataset_dir, loras_dir, llama_model)
        elif stage == "dpo":
            from .train_dpo import train_dpo
            train_dpo(self, name, memories, options, dataset_dir, loras_dir, llama_model)
        else:
            raise ValueError(f"Unsupported stage: {stage}")
