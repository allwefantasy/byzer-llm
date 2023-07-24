from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import transformers
import torch
from typing import Dict,List,Tuple
from byzerllm.utils import generate_instruction_from_history,compute_max_new_tokens,tokenize_stopping_sequences_where_needed
import os
import time



def stream_chat(self,tokenizer,ins:str, his:List[Dict[str,str]]=[],  
        max_length:int=4090, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    timeout_s = float(kwargs.get("timeout_s",60*5))
    
    stopping_sequences = []
    
    if "stopping_sequences" in kwargs:        
        stopping_sequences = tokenize_stopping_sequences_where_needed(tokenizer,kwargs["stopping_sequences"].split(","))
    
    role_mapping = {        
        "user":"User",        
        "assistant":"Assistant",
    }
    
    fin_ins = generate_instruction_from_history(ins,his,role_mapping=role_mapping)     

    tokens = tokenizer(fin_ins, return_token_type_ids=False,return_tensors="pt").to(device)
    
    max_new_tokens = compute_max_new_tokens(tokens,max_length)    
    start_timestamp = time.time()
    response = self.generate(
        input_ids=tokens["input_ids"],
        max_new_tokens= max_new_tokens,
        repetition_penalty=1.05,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
        max_time_criteria=(timeout_s,start_timestamp),
        stopping_sequences=stopping_sequences,
    )
    answer = tokenizer.decode(response[0][tokens["input_ids"].shape[1]:], skip_special_tokens=True)
    return [(answer,"")]


def init_model(model_dir,infer_params:Dict[str,str]={},sys_conf:Dict[str,str]={}):
    longContextMode = infer_params.get("longContextMode","true") == "true"

    if longContextMode:
        old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__
        def ntk_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):

            #The method is just these three lines
            max_position_embeddings = 16384
            a = 8 #Alpha value
            base = base * a ** (dim / (dim-2)) #Base change formula

            old_init(self, dim, max_position_embeddings, base, device)    
        
        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = ntk_scaled_init

    pretrained_model_dir = os.path.join(model_dir,"pretrained_model")
    adaptor_model_dir = model_dir
    is_adaptor_model = os.path.exists(pretrained_model_dir)

    if not is_adaptor_model:        
        pretrained_model_dir = model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    tokenizer.padding_side="right"
    tokenizer.pad_token_id=0
    
    quatization = infer_params.get("quatization","false") == "true"

    if quatization:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_dir,
            trust_remote_code=True,            
            device_map="auto",
            quantization_config=nf4_config,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir,trust_remote_code=True,
                                                device_map='auto',                                                
                                                torch_dtype=torch.bfloat16                                                
                                                )
    if is_adaptor_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adaptor_model_dir)

    model.eval()  
    if quatization:
        model = torch.compile(model)      
    import types
    model.stream_chat = types.MethodType(stream_chat, model)     
    return (model,tokenizer)


