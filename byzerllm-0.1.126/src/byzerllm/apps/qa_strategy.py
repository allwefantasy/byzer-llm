import io

from langchain.docstore.document import Document
from abc import ABC, abstractmethod
from typing import List, Tuple,Dict,Any
import json
import csv


class DocRetrieveStrategy(ABC):
    @abstractmethod
    def retrieve(self, docs: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        pass


class FullDocRetrieveStrategy(DocRetrieveStrategy):
    def retrieve(self, docs: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        if not docs:            
            return []

        doc_hits = {}
        for doc in docs:
            source = doc[0].metadata['source']            
            doc,score,hits = doc_hits.get(source, (doc[0], doc[1], 0))
            doc_hits[source] = (doc,score,hits+1)

        # Sort by hits descending and then by score ascending
        sorted_docs = sorted(doc_hits.values(), key=lambda x: (x[2], -1 * x[1]), reverse=True)
        doc,score,hits = sorted_docs[0]
        doc.page_content = doc.metadata['page_content']
        return [(doc, score)]


class DocRetrieveStrategyFactory(DocRetrieveStrategy):
    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def retrieve(self, docs: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        if self.strategy == "full_doc":
            print("Using full_doc strategy",flush=True)
            return FullDocRetrieveStrategy().retrieve(docs, k)
        else:
            return docs[0:k]


class DocCombineFormat(ABC):
    @abstractmethod
    def combine(self, docs: List[Tuple[Document, float]], k: int) -> Tuple[str, List[Dict[Any, Any]]]:
        pass


class FullDocCombineFormatList(DocCombineFormat):
    def __init__(self, input: Dict[str,Any]) -> None:
        self.params = input

    def combine(self, docs: List[Tuple[Document, float]], k: int) -> Tuple[str, List[Dict[Any, Any]]]:
        if not docs :
            return None

        temp_docs = []
        temp_metas = []
        for index, doc in enumerate(docs[0:k]):
            temp_docs.append(f'{index}. {doc[0].page_content}')
            if "metadata" in doc[0]:
                temp_metas.append(doc[0].metadata)
        return ("\n".join(temp_docs), temp_metas)


class FullDocCombineFormatDefault(DocCombineFormat):
    def __init__(self, input: Dict[str,Any]) -> None:
        self.params = input

    def combine(self, docs: List[Tuple[Document, float]], k: int) -> Tuple[str, List[Dict[Any, Any]]]:
        if docs is None or len(docs) == 0:
            return None

        temp_docs = []
        temp_metas = []
        for index, doc in enumerate(docs[0:k]):
            temp_docs.append(f'{doc[0].page_content}')
            if "metadata" in doc[0]:
                temp_metas.append(doc[0].metadata)
        return ("\n".join(temp_docs), temp_metas)

class JsonCombineFormat(DocCombineFormat):
    def __init__(self, input: Dict[str,Any]) -> None:
        self.params = input

    def combine(self, docs: List[Tuple[Document, float]], k: int) -> Tuple[str, List[Dict[Any, Any]]]:
        if docs is None or len(docs) == 0:
            return None

        temp_docs = []
        temp_metas = []
        for index, doc in enumerate(docs[0:k]):
            temp_docs.append({"body":doc[0].page_content})
            if "metadata" in doc[0]:
                temp_metas.append(doc[0].metadata)
        return (json.dumps(temp_docs,ensure_ascii=False), temp_metas)


class CsvCombineFormat(DocCombineFormat):
    def __init__(self, input: Dict[str,Any]) -> None:
        self.params = input

    def combine(self, docs: List[Tuple[Document, float]], k: int) -> Tuple[str, List[Dict[Any, Any]]]:
        if docs is None or len(docs) == 0:
            return None
                
        with io.StringIO() as csv_buffer:
            csv_writer = csv.DictWriter(csv_buffer, fieldnames=['body'])
            csv_writer.writeheader()
            temp_metas = []
            for index, doc in enumerate(docs[0:k]):
                csv_writer.writerow({"body": doc[0].page_content})
                if "metadata" in doc[0]:
                    temp_metas.append(doc[0].metadata)

            value = csv_buffer.getvalue()        
        return (value, temp_metas)


class FullDocCombineFormatFactory(DocCombineFormat):
    def __init__(self, input: Dict[str,Any]) -> None:
        self.params = input
        self.format = input.get("format","")       

    def combine(self, docs: List[Tuple[Document, float]], k: int) -> Tuple[str, List[Dict[Any, Any]]]:
        if self.format == 'list':
            return FullDocCombineFormatList(self.params).combine(docs, k)
        elif self.format == 'json':
            return JsonCombineFormat(self.params).combine(docs, k)
        elif self.format == 'csv':
            return CsvCombineFormat(self.params).combine(docs, k)
        else:
            return FullDocCombineFormatDefault(self.params).combine(docs, k)
