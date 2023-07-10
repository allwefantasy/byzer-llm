from typing import List,Tuple,Any,Dict
import json
from .emb import ByzerLLMEmbeddings,ByzerSentenceTransformerEmbeddings

class ByzerLLMGenerator:
    def __init__(self,model,tokenizer,use_feature_extraction=False) -> None:
        self.model = model        
        self.embedding = None
        self.tokenizer = None
        if tokenizer:
            self.tokenizer = tokenizer
            from sentence_transformers import SentenceTransformer
            if isinstance(model, SentenceTransformer) or isinstance(self.tokenizer, SentenceTransformer):
                self.embedding = ByzerSentenceTransformerEmbeddings(model,tokenizer)
            else:    
                self.embedding = ByzerLLMEmbeddings(model,tokenizer,use_feature_extraction=use_feature_extraction)
    
    def extract_history(self,input)-> List[Tuple[str,str]]:
        history = input.get("history",[])
        return [(item["query"],item["response"]) for item in history]
    
    def predict(self,query:Dict[str,Any]):
        ins = query["instruction"]
        if query.get("embedding",False):
            if not self.embedding:
                raise Exception("This model do not support text emedding service")
            return self.embedding.embed_query(ins)
        
        if not self.model:
            raise Exception("This model do not support text generation service")

        his = self.extract_history(query) 

        new_params = {}
        
        if "image" in query:
            new_params["image"] = query["image"] 

        for p in ["inference_mode"] :
            if p in query:
                new_params[p] = query[p]
            
        response = self.model.stream_chat(self.tokenizer, 
        ins, his, 
        max_length=int(query.get("max_length",1024)), 
        top_p=float(query.get("top_p",0.7)),
        temperature=float(query.get("temperature",0.9)),**new_params)
        
        last = ""
        for t,_ in response:                                               
            last=t        
        return last    


def simple_predict_func(model,v):
    (model,tokenizer) = model
    llm = ByzerLLMGenerator(model,tokenizer)
    data = [json.loads(item) for item in v]
    
    results=[]
    for item in data:        
        v = llm.predict(item)
        results.append({
            "predict":v,
            "input":item})

    return {"value":[json.dumps(results,ensure_ascii=False)]}


def chatglm_predict_func(model,v):
    (trainer,tokenizer) = model
    llm = ByzerLLMGenerator(trainer,tokenizer,use_feature_extraction=True)
    data = [json.loads(item) for item in v]
    
    results=[]
    for item in data:
        if "system" in item:
            item["instruction"] = f'{item["system"]}\n{item["instruction"]}'
        v = llm.predict(item)
        results.append({
            "predict":v,
            "input":item})
        
    return {"value":[json.dumps(results,ensure_ascii=False)]}

def qa_predict_func(model,v):        
    data = [json.loads(item) for item in v]
    
    results=[]
    for item in data:
        if "system" in item:
            item["instruction"] = f'{item["system"]}\n{item["instruction"]}'
        v = model.predict(item)
        results.append({
            "predict":v,
            "input":item})
        
    return {"value":[json.dumps(results,ensure_ascii=False)]}