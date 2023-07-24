from pathlib import Path
from typing import Any, TypeVar, Dict,Union,List
import torch
from transformers import PreTrainedTokenizer

T = TypeVar("T")

def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

import signal
from contextlib import contextmanager
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

@contextmanager
def timeout(duration: float):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

def generate_instruction_from_history(ins:str,his:List[Dict[str,str]],role_mapping:Dict[str,str]={        
        "user":"User",        
        "assistant":"Assistant",
    }):

    new_his = []    
    for item in his:
        if item["role"] == "system":
            new_his.append(item["content"])
            continue        
        new_his.append(f"{role_mapping[item['role']]}:{item['content']}")            

    # here we should make sure the user build the conversation string manually also
    # works. This means if the user do not provide  the history, then
    # we should treat ins as conversation string which the user build manually
    if len(new_his) > 0 and ins != "":
        new_his.append(f"{role_mapping['user']}:{ins}")
        new_his.append(f"{role_mapping['assistant']}:")

    if len(new_his) > 0 and ins == "":
        new_his.append(f"{role_mapping['assistant']}:")            
    
    if len(new_his) == 0:
        new_his.append(ins)    

    fin_ins = "\n".join(new_his)
    return fin_ins  

def compute_max_new_tokens(tokens,max_length:int):
    input_length = tokens["input_ids"].shape[1]
    max_new_tokens = max_length - input_length
    if max_new_tokens <= 0:
        raise Exception(f"Input is too long ({input_length}). Try to reduce the length of history or use a larger `max_length` value (now:{max_length})")
    return max_new_tokens

def tokenize_string(tokenizer: PreTrainedTokenizer, key: str) -> Union[int, List[int]]:
    """Tokenize a string using a tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        key (str): String to tokenize.
    """
    token_ids = tokenizer.encode(key, add_special_tokens=False)
    return token_ids[0] if len(token_ids) == 1 else token_ids

def tokenize_stopping_sequences_where_needed(
    tokenizer: PreTrainedTokenizer,
    stopping_sequences: List[Union[str, int, List[int]]],
) -> List[Union[List[int], int]]:
    """If any sequence is a string, tokenize it.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        stopping_sequences (List[Union[str, int, List[int]]]): Stopping sequences to
            tokenize. Can be ids, sequences of ids or strings.
    """
    if not stopping_sequences:
        return None
    return [
        tokenize_string(tokenizer, sequence) if isinstance(sequence, str) else sequence
        for sequence in stopping_sequences
    ]

