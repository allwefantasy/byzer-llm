from langchain.embeddings.base import Embeddings
from typing import List

class ByzerLLMEmbeddings(Embeddings):
    def __init__(self, model,tokenizer): 
        from transformers import pipeline              
        self.pipeline = pipeline("feature-extraction", model = model, tokenizer = tokenizer,device=0)
        
    def _encode(self,text:str):
        return self.pipeline(text)[0][-1]
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:        
        embeddings = [self._encode(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:    
        embedding = self._encode(text)
        return embedding