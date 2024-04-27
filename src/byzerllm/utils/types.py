import time
import threading
from typing import TYPE_CHECKING,TypeVar,Dict, List, Optional, Union,Any,Tuple,get_type_hints,Annotated,get_args,Callable
from queue import Queue
from transformers import StoppingCriteria

class StopSequencesCriteria(StoppingCriteria):
    import torch
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

class SingleOutputMeta:
    def __init__(self, input_tokens_count:int=0, generated_tokens_count:int=0):        
        self.input_tokens_count = input_tokens_count
        self.generated_tokens_count = generated_tokens_count    

class SingleOutput:
    def __init__(self, text:str,metadata:SingleOutputMeta=SingleOutputMeta()):
        self.text = text
        self.metadata = metadata
        
class StreamOutputs: 
    def __init__(self, outputs:List[SingleOutput]):
        self.outputs = outputs   

class BlockBinaryStreamServer:
    def __init__(self):
        self.cache = {}
        self.cache_status = {} 
        self.lock = threading.Lock()

    def add_item(self, request_id, item):
        with self.lock:            
            if request_id not in self.cache:
                self.cache[request_id] = Queue()
            self.cache[request_id].put(item)
            self.cache_status[request_id]=int(time.time()*1000)
    
    def mark_done(self, request_id):
        if len(self.cache_status) > 30:
            now = int(time.time()*1000)
            with self.lock:
                for k in list(self.cache_status.keys()):
                    if now - self.cache_status[k] > 10*60*60*1000:
                        del self.cache_status[k]
                        del self.cache[k] 
        with self.lock:            
            self.cache_status[request_id] = 0

    def get_item(self, request_id):                
        with self.lock:
            if request_id not in self.cache:
                return None                                                        
            try:
                return self.cache[request_id].get(timeout=0.1)                
            except:
                if request_id in self.cache_status and self.cache_status[request_id] == 0:
                    del self.cache[request_id]
                    del self.cache_status[request_id]
                    return None
                return "RUNNING"


class BlockVLLMStreamServer:
    def __init__(self):
        self.cache = {}
        self.cache_status = {} 
        self.lock = threading.Lock()

    def add_item(self, request_id, item):
        with self.lock:            
            self.cache[request_id]=item
            self.cache_status[request_id]=int(time.time()*1000)
    
    def mark_done(self, request_id):
        if len(self.cache_status) > 30:
            now = int(time.time()*1000)
            with self.lock:
                for k in list(self.cache_status.keys()):
                    if now - self.cache_status[k] > 10*60*60*1000:
                        del self.cache_status[k]
                        del self.cache[k] 
        with self.lock:            
            self.cache_status[request_id] = 0

    def get_item(self, request_id):                
        with self.lock:
            v = self.cache.get(request_id, None)     
            if request_id in self.cache_status and self.cache_status[request_id] == 0:
                del self.cache[request_id]
                del self.cache_status[request_id]
            return v     

class VLLMStreamServer:
    def __init__(self):
        self.cache = {}
        self.cache_status = {} 
        self.lock = threading.Lock()

    async def add_item(self, request_id, item):
        with self.lock:            
            self.cache[request_id]=item
            self.cache_status[request_id]=int(time.time()*1000)
    
    async def mark_done(self, request_id):
        if len(self.cache_status) > 30:
            now = int(time.time()*1000)
            with self.lock:
                for k in list(self.cache_status.keys()):
                    if now - self.cache_status[k] > 10*60*60*1000:
                        del self.cache_status[k]
                        del self.cache[k] 
        with self.lock:            
            self.cache_status[request_id] = 0

    async def get_item(self, request_id):                
        with self.lock:
            v = self.cache.get(request_id, None)     
            if request_id in self.cache_status and self.cache_status[request_id] == 0:
                del self.cache[request_id]
                del self.cache_status[request_id]
            return v    