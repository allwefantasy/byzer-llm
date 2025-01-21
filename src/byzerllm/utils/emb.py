from typing import List, Union

"""**Embeddings** interface."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.runnables.config import run_in_executor


class Embeddings(ABC):
    """An interface for embedding models.

    This is an interface meant for implementing text embedding models.

    Text embedding models are used to map text to a vector (a point in n-dimensional
    space).

    Texts that are similar will usually be mapped to points that are close to each
    other in this space. The exact details of what's considered "similar" and how
    "distance" is measured in this space are dependent on the specific embedding model.

    This abstraction contains a method for embedding a list of documents and a method
    for embedding a query text. The embedding of a query text is expected to be a single
    vector, while the embedding of a list of documents is expected to be a list of
    vectors.

    Usually the query embedding is identical to the document embedding, but the
    abstraction allows treating them independently.

    In addition to the synchronous methods, this interface also provides asynchronous
    versions of the methods.

    By default, the asynchronous methods are implemented using the synchronous methods;
    however, implementations may choose to override the asynchronous methods with
    an async native implementation for performance reasons.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)


try:
    import torch
    import torch.nn.functional as F
    from transformers import pipeline

    class ByzerSentenceTransformerEmbeddings(Embeddings):

        def __init__(self, model, tokenizer, device="auto"):
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

        def _encode(self, texts: List[str], extract_params={}):
            embeddings = [
                emb.tolist() for emb in self.model.encode(texts, **extract_params)
            ]
            return embeddings

        def embed_documents(
            self, texts: List[str], extract_params={}
        ) -> List[List[float]]:
            embeddings = self._encode(texts, extract_params)
            return embeddings

        def embed_query(self, text: str, extract_params={}) -> List[float]:
            embedding = self._encode([text], extract_params)
            return embedding[0]

    class ByzerLLMEmbeddings(Embeddings):
        def __init__(
            self, model, tokenizer, device="auto", use_feature_extraction=False
        ):
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
                self.pipeline = pipeline(
                    "feature-extraction", model=model, tokenizer=tokenizer, device=0
                )

        def _encode(self, texts: List[str], extract_params={}):
            if self.pipeline:
                return [self.pipeline(text)[0][-1] for text in texts]
            else:
                _, embeddings = self.get_embedding_with_token_count(texts)
                embeddings = embeddings.detach().cpu()
                embeddings = embeddings.numpy()
                embeddings = [emb.tolist() for emb in embeddings]
                return embeddings

        def embed_documents(
            self, texts: List[str], extract_params={}
        ) -> List[List[float]]:
            embeddings = self._encode(texts, extract_params)
            return embeddings

        def embed_query(self, text: str, extract_params={}) -> List[float]:
            embedding = self._encode([text], extract_params)
            return embedding[0]

        # copied from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#usage-huggingface-transformers
        def get_embedding_with_token_count(
            self,
            sentences: Union[str, List[str]],
            ignore_keys: List[str] = ["token_type_ids"],
        ):
            # Mean Pooling - Take attention mask into account for correct averaging
            def mean_pooling(model_output, attention_mask):
                # First element of model_output contains all token embeddings
                token_embeddings = model_output[0]
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                return torch.sum(
                    token_embeddings * input_mask_expanded, 1
                ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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

except ImportError:

    class ByzerLLMEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

        def embed_documents(self, *args, **kwargs):
            raise ImportError("transformers is not installed")

        def embed_query(self, *args, **kwargs):
            raise ImportError("transformers is not installed")

    class ByzerSentenceTransformerEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

        def embed_documents(self, *args, **kwargs):
            raise ImportError("transformers is not installed")

        def embed_query(self, *args, **kwargs):
            raise ImportError("transformers is not installed")
