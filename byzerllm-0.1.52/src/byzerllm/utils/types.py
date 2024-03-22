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