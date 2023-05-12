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

# 加载文件夹中的所有txt类型的文件
loader = DirectoryLoader('/home/winubuntu/projects/byzer-doc/byzer-lang/zh-cn', glob='**/*.md')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()

# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=30)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

# 初始化 openai 的 embeddings 对象
embeddings = LocalHuggingFaceEmbeddings('multi-qa-mpnet-base-dot-v1')
# 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
# docsearch = Chroma.from_documents(split_docs, embeddings)
FAISS_INDEX_PATH="/my8t/byzerllm/tmp/faiss_index"
db = FAISS.from_documents(split_docs, embeddings)
db.save_local(FAISS_INDEX_PATH)

# 创建问答对象

gloabl_prompt="请尝试从下面内容中学习Byzer相关知识："
result = db.similarity_search_with_score("如何使用 Byzer 加载csv 格式数据？")
prompt = "".join([doc[0].page_content for doc in result])
chat((gloabl_prompt + prompt)[0:400]+"。 现在，请问我们应该如何使用 Byzer 加载csv 格式数据？",[])