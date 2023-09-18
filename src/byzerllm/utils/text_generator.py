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
    
    def extract_history(self,input)-> List[Dict[str,str]]:
        history = input.get("history",[])
        return history
    
    def predict(self,query:Dict[str,Any]):
        ins = query["instruction"]
        if query.get("embedding",False):
            if not self.embedding:
                raise Exception("This model do not support text emedding service")
            new_params = {}
            for k,v in query.items():
                if k.startswith("gen."):
                    new_params[k[len("gen."):]] = v
                if k.startswith("generation."):
                    new_params[k[len("generation."):]] = v 
            
            if hasattr(self.embedding.model,"embed_query"):
                return self.embedding.model.embed_query(ins,extract_params=new_params)
            
            return self.embedding.embed_query(ins,extract_params=new_params)
        
        if not self.model:
            raise Exception("This model do not support text generation service")

        his = self.extract_history(query) 
        
        # notice that not all parameters in query are used in model stream_chat function
        # only the following parameters and the name starts with "gen." or "generation." are used
        # the prefix "gen." or "generation." will be removed when passing to model stream_chat function
        new_params = {}
        
        if "image" in query:
            new_params["image"] = query["image"] 
        
        for p in ["inference_mode","stopping_sequences","timeout_s","stopping_sequences_skip_check_min_length"]:
            if p in query:
                new_params[p] = query[p]

        for k,v in query.items():
            if k.startswith("gen."):
                new_params[k[len("gen."):]] = v
            if k.startswith("generation."):
                new_params[k[len("generation."):]] = v     

        response = self.model.stream_chat(self.tokenizer, 
        ins, his, 
        max_length=int(query.get("max_length",1024)), 
        top_p=float(query.get("top_p",0.7)),
        temperature=float(query.get("temperature",0.9)),**new_params)
        
        last = ""
        for t,_ in response:                                               
            last=t        
        return last  

    async def async_predict(self,query:Dict[str,Any]):
            ins = query["instruction"]
            if query.get("embedding",False):
                if not self.embedding:
                    raise Exception("This model do not support text emedding service")
                new_params = {}
                for k,v in query.items():
                    if k.startswith("gen."):
                        new_params[k[len("gen."):]] = v
                    if k.startswith("generation."):
                        new_params[k[len("generation."):]] = v 
                
                if hasattr(self.embedding.model,"embed_query"):
                    return self.embedding.model.embed_query(ins,extract_params=new_params)
                
                return self.embedding.embed_query(ins,extract_params=new_params)
            
            if not self.model:
                raise Exception("This model do not support text generation service")

            his = self.extract_history(query) 
            
            # notice that not all parameters in query are used in model stream_chat function
            # only the following parameters and the name starts with "gen." or "generation." are used
            # the prefix "gen." or "generation." will be removed when passing to model stream_chat function
            new_params = {}
            
            if "image" in query:
                new_params["image"] = query["image"] 
            
            for p in ["inference_mode","stopping_sequences","timeout_s","stopping_sequences_skip_check_min_length"]:
                if p in query:
                    new_params[p] = query[p]

            for k,v in query.items():
                if k.startswith("gen."):
                    new_params[k[len("gen."):]] = v
                if k.startswith("generation."):
                    new_params[k[len("generation."):]] = v     
            if hasattr(self.model, "async_stream_chat"):
                response = await self.model.async_stream_chat(self.tokenizer, 
                ins, his, 
                max_length=int(query.get("max_length",1024)), 
                top_p=float(query.get("top_p",0.7)),
                temperature=float(query.get("temperature",0.9)),**new_params)
            else:
                response = self.model.stream_chat(self.tokenizer, 
                ins, his, 
                max_length=int(query.get("max_length",1024)), 
                top_p=float(query.get("top_p",0.7)),
                temperature=float(query.get("temperature",0.9)),**new_params)    
            
            last = ""
            for t,_ in response:                                               
                last=t        
            return last        


async def simple_predict_func(model,v):
    (model,tokenizer) = model
    llm = ByzerLLMGenerator(model,tokenizer)
    data = [json.loads(item) for item in v]
    
    results=[]
    for item in data:        
        v = await llm.async_predict(item)
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