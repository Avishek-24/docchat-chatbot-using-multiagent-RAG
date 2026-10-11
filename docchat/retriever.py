"""Hybrid retrieval: BM25 + persistent Chroma semantic search, merged with RRF.
A NumPy cosine fallback keeps tests and offline recovery possible.

If the embedding model can't load (e.g. you're offline on first run), the
retriever quietly falls back to keyword-only search instead of crashing.
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
try:
    from rank_bm25 import BM25Plus
except ImportError:  # lightweight fallback for offline smoke tests
    class BM25Plus:
        """Small positive keyword scorer used only when rank-bm25 is unavailable."""
        def __init__(self, corpus):
            self.corpus = corpus
            self.doc_freq = {}
            for doc in corpus:
                for token in set(doc):
                    self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        def get_scores(self, query_tokens):
            n = max(1, len(self.corpus))
            scores = []
            for doc in self.corpus:
                counts = {}
                for token in doc:
                    counts[token] = counts.get(token, 0) + 1
                score = 0.0
                for token in query_tokens:
                    if token in counts:
                        idf = np.log(1 + (n - self.doc_freq.get(token, 0) + 0.5) /
                                     (self.doc_freq.get(token, 0) + 0.5))
                        score += float(idf) * counts[token] / (counts[token] + 1.2)
                scores.append(score)
            return np.asarray(scores, dtype="float32")

from .loader import Chunk

RRF_K = 60
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # ~130 MB, runs on CPU via ONNX
CACHE_DIR = str(Path(__file__).resolve().parent.parent / ".cache" / "fastembed")
CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "chroma")


def tokenize(text: str) -> List[str]:
    # keeps decimals like "1.21" as one token, which matters for numeric tables
    return re.findall(r"\w+(?:\.\d+)?", text.lower())


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding

        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(model_name=model_name, cache_dir=CACHE_DIR)

    def encode(self, texts: List[str], batch_size: int = 32,
               progress: Optional[Callable[[float], None]] = None) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            out.extend(np.asarray(v, dtype="float32") for v in self.model.embed(batch))
            if progress:
                progress(min(i + batch_size, len(texts)) / len(texts))
        arr = np.vstack(out)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


@dataclass
class Hit:
    chunk: Chunk
    score: float            # fused (RRF) score
    dense: Optional[float]  # cosine similarity, if embeddings are available
    bm25: float


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], embedder: Optional[Embedder] = None,
                 progress: Optional[Callable[[float], None]] = None):
        if not chunks:
            raise ValueError("Cannot build a retriever without chunks.")
        self.chunks = chunks
        self.embedder = embedder
        self.vectors: Optional[np.ndarray] = None
        self.dense_error: Optional[str] = None
        self.collection = None
        self.chroma_error: Optional[str] = None

        corpus = [tokenize(c.text) or ["_empty_"] for c in chunks]
        self.bm25 = BM25Plus(corpus)  # unlike BM25Okapi, stays positive on tiny corpora

        if embedder is not None:
            try:
                texts = [c.text for c in chunks]
                self.vectors = embedder.encode(texts, progress=progress)
                # Chroma is the persistent vector store. Each distinct document
                # set receives its own collection so uploads cannot contaminate each other.
                try:
                    import chromadb
                    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
                    fingerprint = hashlib.sha256(
                        "\\n".join(f"{c.source}|{c.page}|{c.kind}|{c.text}" for c in chunks).encode("utf-8")
                    ).hexdigest()[:20]
                    client = chromadb.PersistentClient(path=CHROMA_DIR)
                    self.collection = client.get_or_create_collection(
                        name=f"docs_{fingerprint}",
                        metadata={"hnsw:space": "cosine", "project": "docchat-local-rag"},
                    )
                    if self.collection.count() == 0:
                        self.collection.add(
                            ids=[str(i) for i in range(len(chunks))],
                            documents=texts,
                            embeddings=self.vectors.tolist(),
                            metadatas=[{"source": c.source, "page": c.page or 0, "kind": c.kind}
                                       for c in chunks],
                        )
                except Exception as exc:
                    self.chroma_error = str(exc)
            except Exception as exc:  # fall back to keyword-only
                self.vectors = None
                self.dense_error = str(exc)

    @property
    def mode(self) -> str:
        if self.vectors is None:
            return "keyword-only"
        return "hybrid (BM25 + Chroma vectors)" if self.collection is not None else "hybrid (BM25 + in-memory vectors)"

    def search(self, query: str, k: int = 6, depth: int = 20) -> List[Hit]:
        n = len(self.chunks)
        depth = min(max(depth, k), n)
        fused: dict = {}

        tokens = tokenize(query)
        bm = self.bm25.get_scores(tokens) if tokens else np.zeros(n)
        for rank, i in enumerate(int(x) for x in np.argsort(-bm)[:depth] if bm[x] > 0):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

        sims = None
        if self.vectors is not None and self.embedder is not None:
            try:
                qv = self.embedder.encode([query])[0]
                sims = self.vectors @ qv
                if self.collection is not None:
                    result = self.collection.query(
                        query_embeddings=[qv.tolist()],
                        n_results=min(depth, len(self.chunks)),
                        include=["distances", "metadatas"],
                    )
                    ids = result.get("ids", [[]])[0]
                    distances = result.get("distances", [[]])[0]
                    dense_ranked = [(int(i), 1.0 - float(d)) for i, d in zip(ids, distances)]
                    for rank, (i, similarity) in enumerate(dense_ranked):
                        sims[i] = similarity
                        fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
                else:
                    for rank, i in enumerate(int(x) for x in np.argsort(-sims)[:depth]):
                        fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            except Exception:
                # If Chroma fails after indexing, retain the safe in-memory dense path.
                for rank, i in enumerate(int(x) for x in np.argsort(-sims)[:depth]):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            Hit(self.chunks[i], score,
                float(sims[i]) if sims is not None else None, float(bm[i]))
            for i, score in top
        ]
