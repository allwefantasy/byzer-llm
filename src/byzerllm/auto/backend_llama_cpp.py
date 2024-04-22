from llama_cpp import Llama
from typing import Any, Dict, List, Tuple, Union

class LlamaCppBackend:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_parts: int = -1,
        seed: int = 1337,
        f16_kv: bool = False,
        logits_all: bool = False,
        vocab_only: bool = False,
        use_mlock: bool = False,
        n_threads: int = 4,
        n_batch: int = 8,
        last_n_tokens_size: int = 64,
        use_mmap: bool = True,
        use_cache: bool = False,
        embedding: bool = False,
        n_gpu_layers: int = -1,
        low_vram: int = 0,
        verbose: bool = False,
        lora_base: str = "",
        lora_path: str = "",
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_parts = n_parts
        self.seed = seed
        self.f16_kv = f16_kv
        self.logits_all = logits_all
        self.vocab_only = vocab_only
        self.use_mlock = use_mlock
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.last_n_tokens_size = last_n_tokens_size
        self.use_mmap = use_mmap
        self.use_cache = use_cache
        self.embedding = embedding
        self.n_gpu_layers = n_gpu_layers
        self.low_vram = low_vram
        self.verbose = verbose
        self.lora_base = lora_base
        self.lora_path = lora_path

        self.llm = None

    def load(self):
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_parts=self.n_parts,
            seed=self.seed,
            f16_kv=self.f16_kv,
            logits_all=self.logits_all,
            vocab_only=self.vocab_only,
            use_mlock=self.use_mlock,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            last_n_tokens_size=self.last_n_tokens_size,
            use_mmap=self.use_mmap,
            use_cache=self.use_cache,
            embedding=self.embedding,
            n_gpu_layers=self.n_gpu_layers,
            low_vram=self.low_vram,
            verbose=self.verbose,
            lora_base=self.lora_base,
            lora_path=self.lora_path
        )

    def generate(
        self,
        prompts: Union[str, List[str]],
        stop: Union[str, List[str]] = [],
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = -1
    ) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            return self.llm(
                prompt=prompts,
                max_tokens=max_tokens,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
                seed=seed
            )["choices"][0]["text"]
        else:
            return [
                self.llm(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed
                )["choices"][0]["text"]
                for prompt in prompts
            ]

    def embed(self, text: Union[List[str], str]) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        embeddings = []
        for t in text:
            output = self.llm.create_embedding(t)
            embedding = output["data"][0]["embedding"]
            embeddings.append(embedding)
        return embeddings