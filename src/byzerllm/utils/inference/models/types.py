from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Optional
from dataclasses import dataclass

class ClientPramaeters(BaseModel):    
    # Activate logits sampling
    do_sample: bool = False
    # Maximum number of generated tokens
    max_new_tokens: int = 20
    # The parameter for repetition penalty. 1.0 means no penalty.
    # See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    repetition_penalty: Optional[float] = None
    # Whether to prepend the prompt to the generated text
    return_full_text: bool = False
    # Stop generating tokens if a member of `stop_sequences` is generated
    stop: List[str] = []
    # Random sampling seed
    seed: Optional[int]
    # The value used to module the logits distribution.
    temperature: Optional[float]
    # The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: Optional[int]
    # If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
    # higher are kept for generation.
    top_p: Optional[float]
    # truncate inputs tokens to the given size
    truncate: Optional[int]
    # Typical Decoding mass
    # See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information
    typical_p: Optional[float]
    # Generate best_of sequences and return the one if the highest token logprobs
    best_of: Optional[int]
    # Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
    watermark: bool = False
    # Get generation details
    details: bool = False
    # Get decoder input token logprobs and ids
    decoder_input_details: bool = False

class ClientRequest(BaseModel):
    # Prompt
    inputs: str
    # Generation parameters
    parameters: Optional[ClientPramaeters]
    # Whether to stream output tokens
    stream: bool = False


class ClientInputToken(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    # Optional since the logprob of the first token cannot be computed
    logprob: Optional[float]

# Generated tokens

class ClientToken(BaseModel):
    # Token ID from the model tokenizer
    id: int
    # Token text
    text: str
    # Logprob
    logprob: float
    # Is the token a special token
    # Can be used to ignore tokens when concatenating
    special: bool


# Generation finish reason

class ClientFinishReason(BaseModel):
    # number of generated tokens == `max_new_tokens`
    Length = "length"
    # the model generated its end of sequence token
    EndOfSequenceToken = "eos_token"
    # the model generated a text included in `stop_sequences`
    StopSequence = "stop_sequence"


# Additional sequences when using the `best_of` parameter

class ClientBestOfSequence(BaseModel):
    # Generated text
    generated_text: str
    # Generation finish reason
    finish_reason: ClientFinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int]
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[ClientInputToken]
    # Generated tokens
    tokens: List[ClientToken]


# `generate` details
class ClientDetails(BaseModel):
    # Generation finish reason
    finish_reason: ClientFinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int]
    # Decoder input tokens, empty if decoder_input_details is False
    prefill: List[ClientInputToken]
    # Generated tokens
    tokens: List[ClientToken]
    # Additional sequences when using the `best_of` parameter
    best_of_sequences: Optional[List[ClientBestOfSequence]]


# `generate` return value
class ClientResponse(BaseModel):
    # Generated text
    generated_text: str
    # Generation details
    details: ClientDetails


# `generate_stream` details
class ClientStreamDetails(BaseModel):
    # Generation finish reason
    finish_reason: ClientFinishReason
    # Number of generated tokens
    generated_tokens: int
    # Sampling seed if sampling was activated
    seed: Optional[int]


# `generate_stream` return value

class ClientStreamDetails(BaseModel):
    # Generated token
    token: ClientToken
    # Complete generated text
    # Only available when the generation is finished
    generated_text: Optional[str]
    # Generation details
    # Only available when the generation is finished
    details: Optional[ClientStreamDetails]


# Inference API currently deployed model
class ClientDeployedModel(BaseModel):
    model_id: str
    sha: str    
       

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



class Request(BaseModel):
    id: int
    inputs: str
    truncate: int
    parameters: NextTokenChooserParameters
    stopping_parameters: StoppingCriteriaParameters
    prefill_logprobs: bool


class BatchRequest(BaseModel):
    id: int
    requests: List[Request]
    size:int
    max_tokens: int


class CachedBatchRequest(BaseModel):
    id: int
    request_ids: List[int]
    size:int
    max_tokens: int


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

  

