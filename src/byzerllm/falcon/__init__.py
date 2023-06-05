from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from typing import List,Tuple



def stream_chat(self,tokenizer,ins:str, his:List[Tuple[str,str]]=[],  
        max_length:int=4096, 
        top_p:float=0.95,
        temperature:float=0.1):
    pipeline = transformers.pipeline(
                "text-generation",
                model=self,
                tokenizer=tokenizer
               )
    reponses = pipeline( ins,
        max_length=max_length,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    return [(res['generated_text'],"") for res in reponses]


def init_model(model_dir):    
    model = AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)  
    import types
    model.stream_chat = types.MethodType(stream_chat, model)      
    return (model,tokenizer)


