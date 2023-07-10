from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

class Batch(ABC):    

    @abstractmethod
    def filter(self, request_ids: List[int]) -> "Batch":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, batches: List["Batch"]) -> "Batch":
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


@dataclass
class NextTokenChooserParameters:
    temperature: float
    top_k: int
    top_p: float
    typical_p: float
    do_sample: bool
    seed: int
    repetition_penalty: float
    watermark: bool

@dataclass
class StoppingCriteriaParameters:
    max_new_tokens: int
    stop_sequences: List[str]
    ignore_eos_token: bool


@dataclass
class Request:
    id: int
    inputs: str
    truncate: int
    parameters: List[NextTokenChooserParameters]
    stopping_parameters: StoppingCriteriaParameters
    prefill_logprobs: bool

@dataclass
class GeneratedText:
    text: str
    generated_tokens: int
    finish_reason: Optional[str]
    seed: Optional[int]


@dataclass
class PrefillTokens:
    token_ids: List[int]
    logprobs: List[float]
    texts: List[str]    

    def __len__(self):
        return len(self.token_ids)


@dataclass
class Generation:
    request_id: int
    prefill_tokens: Optional[PrefillTokens]
    token_id: int
    token_logprob: float
    token_text: str
    token_is_special: bool
    generated_text: Optional[GeneratedText]

