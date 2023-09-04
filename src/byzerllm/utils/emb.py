from langchain.embeddings.base import Embeddings
from typing import List, Union
import torch
import torch.nn.functional as F
from transformers import pipeline

class ByzerSentenceTransformerEmbeddings(Embeddings):
    
    def __init__(self, model,tokenizer, device="auto"):         
        if model:
            self.model = model
        if tokenizer:
            self.model = tokenizer            
        
        if device == "auto":
            self.device = (
                torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device
        
        
    def _encode(self,texts: List[str],extract_params={}):        
        embeddings = [emb.tolist() for emb in self.model.encode(texts,**extract_params)]
        return embeddings
        
    def embed_documents(self, texts: List[str],extract_params={}) -> List[List[float]]:        
        embeddings = self._encode(texts,extract_params)
        return embeddings

    def embed_query(self, text: str,extract_params={}) -> List[float]:    
        embedding = self._encode([text],extract_params)
        return embedding[0]
        

class ByzerLLMEmbeddings(Embeddings):
    def __init__(self, model,tokenizer, device="auto", use_feature_extraction=False):         
        self.model = model
        self.tokenizer = tokenizer
        
        if device == "auto":
            self.device = (
                torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.pipeline = None
        if use_feature_extraction:
            self.pipeline = pipeline("feature-extraction", model = model, tokenizer = tokenizer,device=0)
        
    def _encode(self,texts: List[str],extract_params={}):
        if self.pipeline:
            return [self.pipeline(text)[0][-1] for text in texts]
        else:
            _, embeddings = self.get_embedding_with_token_count(texts)
            embeddings = embeddings.detach().cpu()
            embeddings = embeddings.numpy()
            embeddings = [emb.tolist() for emb in embeddings]
            return embeddings
        
    def embed_documents(self, texts: List[str],extract_params={}) -> List[List[float]]:        
        embeddings = self._encode(texts,extract_params)
        return embeddings

    def embed_query(self, text: str, extract_params={}) -> List[float]:    
        embedding = self._encode([text],extract_params)
        return embedding[0]

    # copied from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#usage-huggingface-transformers
    def get_embedding_with_token_count(self, sentences: Union[str, List[str]], ignore_keys:List[str]=["token_type_ids"]):
        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            # First element of model_output contains all token embeddings
            token_embeddings = model_output[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = encoded_input.to(self.device)
        token_count = inputs["attention_mask"].sum(dim=1).tolist()[0]
        # Compute token embeddings

        for ignore_key in ignore_keys:
            if hasattr(inputs, ignore_key):
                del inputs[ignore_key]

        model_output = self.model(**inputs)
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, inputs["attention_mask"])
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return token_count, sentence_embeddings