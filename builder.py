from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

class RetrieverBuilder:
    def __init__(self, model_name: str = "mxbai-embed-large"):
        self.embeddings = OllamaEmbeddings(model=model_name)

    def build_hybrid_retriever(self, docs: List[Document], k: int = 8) -> EnsembleRetriever:
        # 1. BM25 Lexical Search
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k

        # 2. ChromaDB Vector Search
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

        # 3. Combine both retrievers using RRF weighting
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        return hybrid_retriever