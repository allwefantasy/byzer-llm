from langchain.docstore.document import Document
from abc import ABC, abstractmethod
from typing import List, Tuple,Dict,Any


class DocRetrieveStrategy(ABC):
    @abstractmethod
    def retrieve(self, docs: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        pass


class FullDocRetrieveStrategy(DocRetrieveStrategy):
    def retrieve(self, docs: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        if docs is None or len(docs) == 0:
            print("empty doc list")
            return None

        doc_hits = {}
        for doc in docs:
            if doc[0].metadata['source'] in doc_hits:
                doc_hits[doc[0].metadata['source']] = (doc[0], doc[1], doc_hits[doc[0].metadata['source']][2] + 1)
            else:
                doc_hits[doc[0].metadata['source']] = (doc[0], doc[1], 1)
        # Sort by hits descending and then by score ascending
        sorted_docs = sorted(doc_hits.values(), key=lambda x: (x[2], -1 * x[1]), reverse=True)
        doc_tuple3 = sorted_docs[0]
        doc_tuple3[0].page_content = doc_tuple3[0].metadata['page_content']
        return [(doc_tuple3[0], doc_tuple3[1])]


class DocRetrieveStrategyFactory(DocRetrieveStrategy):
    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def retrieve(self, docs: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
        if self.strategy == "full_doc":
            print("Using full_doc strategy")
            return FullDocRetrieveStrategy().retrieve(docs, k)
        else:
            return docs


class DocCombineFormat(ABC):
    @abstractmethod
    def combine(self, docs: List[Tuple[Document, float]], k: int) -> Tuple[str, List[Dict[Any, Any]]]:
        pass


class FullDocCombineFormatList(DocCombineFormat):
    def __init__(self, input: Dict[str,Any]) -> None:
        self.format = input.get("format","")

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
        self.format = input.get("format","")

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


class FullDocCombineFormatFactory(DocCombineFormat):
    def __init__(self, input: Dict[str,Any]) -> None:
        self.params = input

    def combine(self, docs: List[Tuple[Document, float]], k: int) -> Tuple[str, List[Dict[Any, Any]]]:
        if self.format == 'list':
            return FullDocCombineFormatList(self.params).combine(docs, k)
        else:
            return FullDocCombineFormatDefault(self.params).combine(docs, k)

