from .generate import InstructionTextGenerationPipeline, load_model_tokenizer_for_generate
from typing import Union, List, Tuple, Optional, Dict
from pyjava.api.mlsql import RayContext,PythonContext
from pyjava.storage import streaming_tar

def restore_model(conf: Dict[str, str],target_dir:str):
    print("restore model...")
    model_servers = RayContext.parse_servers(conf["modelServers"])    
    model_binary = RayContext.collect_from(model_servers)
    streaming_tar.save_rows_as_file(model_binary,target_dir)
    print(f"Restore model done.")

class Inference:
   
   def __init__(self,model_path:str,load_in_8bit:bool= False) -> None:
      self.model_path = model_path
      model, tokenizer = load_model_tokenizer_for_generate(model_path,load_in_8bit=load_in_8bit)
      self.model = model
      self.tokenizer = tokenizer
      self.generation_pipeline = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
   
   def __call__(self, input:str,num_return_sequences:int=2):
      results = self.generation_pipeline(input, num_return_sequences=num_return_sequences)
      return results
