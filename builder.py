from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = "chroma_db"
COLLECTION = "docchat"

# class RetrieverBuilder:
#     def __init__(self, model_name: str = "mxbai-embed-large"):
#         self.embeddings = OllamaEmbeddings(model=model_name)

class RetrieverBuilder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    def _assemble(self, docs: List[Document], vector_store, k: int) -> EnsembleRetriever:
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = k
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],
        )


    def build_hybrid_retriever(self, docs: List[Document], k: int = 8) -> EnsembleRetriever:
        # Replace any previously stored index so old and new chunks don't mix or duplicate
        try:
            Chroma(
                collection_name=COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_DIR,
            ).delete_collection()
        except Exception:
            pass

        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=COLLECTION,
            persist_directory=CHROMA_DIR,      # <-- this is what makes it persistent
        )
        return self._assemble(docs, vector_store, k)

    def load_hybrid_retriever(self, k: int = 8):
        """Rebuild the retriever from the saved index after a restart. Returns None if nothing is saved."""
        vector_store = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )
        data = vector_store.get()
        if not data["documents"]:
            return None
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        return self._assemble(docs, vector_store, k)