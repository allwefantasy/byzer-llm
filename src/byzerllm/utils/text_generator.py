from typing import List,Tuple,Any,Dict
from .emb import ByzerLLMEmbeddings

class ByzerLLMGenerator:
    def __init__(self,model,tokenizer) -> None:
        self.model = model        
        self.embedding = None
        self.tokenizer = None
        if tokenizer:
            self.tokenizer = tokenizer
            self.embedding = ByzerLLMEmbeddings(model,tokenizer)
    
    def extract_history(self,input)-> List[Tuple[str,str]]:
        history = input.get("history",[])
        return [(item["query"],item["response"]) for item in history]
    
    def predict(self,query:Dict[str,Any]):
        ins = query["instruction"]
        if query.get("embedding",False):
            if not self.embedding:
                raise Exception("This model do not support emedding service")
            return self.embedding.embed_query(ins)
        
        his = self.extract_history(query)        

        response = self.model.stream_chat(self.tokenizer, 
        ins, his, 
        max_length=int(query.get("max_length",1024)), 
        top_p=float(query.get("top_p",0.7)),
        temperature=float(query.get("temperature",0.9)))
        
        last = ""
        for t,_ in response:                                               
            last=t        
        return last    


