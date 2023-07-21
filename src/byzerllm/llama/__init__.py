from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from typing import Dict,List,Tuple



def stream_chat(self,tokenizer,ins:str, his:List[Dict[str,str]]=[],  
        max_length:int=4090, 
        top_p:float=0.95,
        temperature:float=0.1,**kwargs):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    role_mapping = {        
        "user":"User",        
        "assistant":"Assistant",
    }
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

    tokens = tokenizer(fin_ins, return_token_type_ids=False,return_tensors="pt").to(device)

    input_length = tokens["input_ids"].shape[1]
    max_new_tokens = max_length - input_length
    if max_new_tokens <= 0:
        raise Exception(f"Input is too long ({input_length}). Try to reduce the length of history or use a larger `max_length` value (now:{max_length})")

    response = self.generate(
        input_ids=tokens["input_ids"],
        max_new_tokens= max_new_tokens,
        repetition_penalty=1.05,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id
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

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side="right"
    tokenizer.pad_token_id=0
    model = AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True,
                                                device_map='auto',                                                
                                                torch_dtype=torch.bfloat16                                                
                                                )
    model.eval()       
    import types
    model.stream_chat = types.MethodType(stream_chat, model)     
    return (model,tokenizer)


