from pathlib import Path
from typing import Any, TypeVar, Dict,Union,List
from functools import wraps
import time
import json
from transformers import PreTrainedTokenizer,StoppingCriteria
import torch

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

def timeit(func):
    """
    Decorator to time a function.
    """

    @wraps(func)
    def inner(*args, **kwargs):
        start_time = time.monotonic()
        ret = func(*args, **kwargs)
        time_taken = time.monotonic() - start_time
        print(f"{func} took {time_taken} s to complete",flush=True)
        return ret

    return inner

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
    return token_ids

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

def  tokenize_stopping_sequences(tokenizer,stop_words):
    stop_words_ids = []
    for stop_word in stop_words:
        w = tokenize_string(tokenizer, stop_word)
        # remove the first token which is empty token 
        # this should work for only llama model
        # if w[0] == 29871 and tokenizer.decode([w[0]],skip_special_tokens=False) == "":
        #     w = w[1:]
        stop_words_ids.append(w)    
    return stop_words_ids

class StopSequencesCriteria(StoppingCriteria):
    """
     skip_check_min_length is used to skip the the stop sequence check if the input_ids is short
     than the min_length. 
    """
    def __init__(self, tokenizer,stops = [],input_start=0, skip_check_min_length=0):
    
      super().__init__()      
      self.stops = stops
      self.input_start = input_start
      self.skip_check_min_length = skip_check_min_length
      self.stop_words= [tokenizer.decode(item,skip_special_tokens=True) for item in stops]
      self.tokenizer = tokenizer   

    def to_str(self,s):
        return self.tokenizer.decode(s,skip_special_tokens=True)     

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):                   
      for index,stop in enumerate(self.stops):                        
        if  self.to_str(input_ids[0][-(len(stop)+10):]).endswith(self.stop_words[index]):
            return True
      return False

def load_json_str(json_str:str):        
    return json.loads(json_str)    


