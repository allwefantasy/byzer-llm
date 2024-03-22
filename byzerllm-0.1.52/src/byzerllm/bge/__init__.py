from sentence_transformers import SentenceTransformer
from typing import Dict,List,Tuple

def _encode(self,texts: List[str],extract_params={}):        
    embeddings = [emb.tolist() for emb in self.encode(texts,
                                                      normalize_embeddings=extract_params.get("normalize_embeddings",True))]
    return embeddings
    
def embed_documents(self, texts: List[str],extract_params={}) -> List[List[float]]:        
    embeddings = self._encode(texts,extract_params)
    return embeddings

def embed_query(self, text: str,extract_params={}) -> List[float]:    
    embedding = self._encode([text],extract_params)
    return embedding[0]
    

def init_model(model_dir,infer_params,sys_conf={}):        
    model = SentenceTransformer(model_dir) 
    import types
    model._encode = types.MethodType(_encode, model) 
    model.embed_documents = types.MethodType(embed_documents, model) 
    model.embed_query = types.MethodType(embed_query, model)     
    return (None,model)


