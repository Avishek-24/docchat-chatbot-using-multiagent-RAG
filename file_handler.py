import hashlib
import pickle
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    def __init__(self, cache_dir: str = ".cache"):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        self.converter = DocumentConverter()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def process(self, file_paths: List[str]) -> List[Document]:
        all_chunks = []
        for path_str in file_paths:
            path = Path(path_str)
            if path.suffix.lower() not in [".pdf", ".docx", ".txt", ".md"]:
                continue

            # Read bytes for cache hashing
            with open(path, "rb") as f:
                content_bytes = f.read()
            file_hash = hashlib.sha256(content_bytes).hexdigest()
            cache_path = self.cache_dir / f"{file_hash}.pkl"

            if cache_path.exists():
                chunks = self._load_from_cache(cache_path)
            else:
                chunks = self._process_file(path_str)
                self._save_to_cache(chunks, cache_path)

            all_chunks.extend(chunks)
        return all_chunks

    def _process_file(self, file_path: str) -> List[Document]:
        result = self.converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        chunks = self.splitter.split_text(markdown_text)
        # Add source file metadata
        for chunk in chunks:
            chunk.metadata["source"] = Path(file_path).name
        return chunks

    def _save_to_cache(self, chunks: List[Document], cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump(chunks, f)

    def _load_from_cache(self, cache_path: Path) -> List[Document]:
        with open(cache_path, "rb") as f:
            return pickle.load(f)