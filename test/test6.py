from byzerllm.utils import ByzerLLMReadTheDocsLoader
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from typing import List
import time
import os
import ray

from langchain.embeddings.base import Embeddings
from typing import List
from sentence_transformers import SentenceTransformer

class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_id): 
        # Should use the GPU by default
        self.model = SentenceTransformer(model_id)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using a locally running
           Hugging Face Sentence Transformer model
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        embeddings =self.model.encode(texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a locally running HF 
        Sentence trnsformer. 
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        embedding = self.model.encode(text)
        return list(map(float, embedding))

FAISS_INDEX_PATH="/my8t/byzerllm/tmp/faiss_index"

loader = ByzerLLMReadTheDocsLoader("/home/winubuntu/projects/byzer-doc/byzer-lang/zh-cn",extension="md")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,
    chunk_overlap  = 20,
    length_function = len,
)

# text_splitter2 = MarkdownTextSplitter(chunk_size=300, chunk_overlap=0)

# Stage one: read all the docs, split them into chunks. 
st = time.time() 
print('Loading documents ...')
docs = loader.load()
#Theoretically, we could use Ray to accelerate this, but it's fast enough as is. 
chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
et = time.time() - st
print(f'Time taken: {et} seconds.') 

# Stage two: embed the docs. 
embeddings = LocalHuggingFaceEmbeddings('multi-qa-mpnet-base-dot-v1')

print(f'Loading chunks into vector store ...') 
st = time.time()
db = FAISS.from_documents(chunks, embeddings)
db.save_local(FAISS_INDEX_PATH)
et = time.time() - st
print(f'Time taken: {et} seconds.')