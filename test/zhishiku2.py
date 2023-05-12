from langchain.embeddings.base import Embeddings

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from typing import Any, List, Mapping, Optional,Tuple
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain import VectorDBQA
from langchain.document_loaders import DirectoryLoader
import requests
import json
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

from byzerllm.chatglm6b.tunning.infer import init_model as init_chatbot_model

def request(sql:str,json_data:str)->str:
    url = 'http://127.0.0.1:9003/model/predict'
    data = {
        'sessionPerUser': 'true',
        'owner': 'william',
        'dataType': 'string',
        'sql': sql,
        'data': json_data
    }
    response = requests.post(url, data=data)
    return response.text

def emb(s:str)-> List[float]:
    json_data = json.dumps([
        {"instruction":s,"embedding":True}
    ])
    response = request('''
     select chat(array(feature)) as value
    ''',json_data)    
    t = json.loads(response)
    t2 = json.loads(t[0]["value"][0])
    return t2[0]["predict"]

def chat(s:str,history:List[Tuple[str,str]])->str:
    newhis = [{"query":item[0],"response":item[1]} for item in history]
    json_data = json.dumps([
        {"instruction":s,"history":newhis,"output":"NAN"}
    ])
    response = request('''
     select chat(array(feature)) as value
    ''',json_data)    
    t = json.loads(response)
    t2 = json.loads(t[0]["value"][0])
    return t2[0]["predict"]


class LocalEmbeddings(Embeddings):
    def __init__(self):
        pass
                
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:        
        embeddings = [emb(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:    
        embedding = emb(text)
        return embedding


class Chatglm6bLLM(LLM):
    
    n: int
        
    @property
    def _llm_type(self) -> str:
        return "chatglm6b"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return chat(prompt,[])
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}




loader = DirectoryLoader('/home/winubuntu/projects/byzer-doc/byzer-lang/zh-cn', glob='**/*.md')

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=30)

split_docs = text_splitter.split_documents(documents)

embeddings = LocalEmbeddings()

FAISS_INDEX_PATH="/my8t/byzerllm/tmp/faiss_index1"
db = FAISS.from_documents(split_docs, embeddings)
db.save_local(FAISS_INDEX_PATH)

qa = VectorDBQA.from_chain_type(llm=Chatglm6bLLM(), chain_type="stuff", vectorstore=db,return_source_documents=True)
result = qa({"query": "科大讯飞今年第一季度收入是多少？"})
print(result)