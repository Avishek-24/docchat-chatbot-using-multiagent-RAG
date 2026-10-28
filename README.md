# DocChat Local RAG — private PDF intelligence assistant

A local-first document question-answering app built with **Streamlit, Ollama, ChromaDB, FastEmbed, and hybrid retrieval**. Upload long reports and ask questions grounded in the source text. Answers include source references and a verification step designed to flag claims that cannot be supported.

## Features

- **Local LLM inference** through Ollama; no hosted LLM API key is required.
- **Persistent Chroma vector database** stored under `data/chroma/`, with a separate collection for each distinct document set.
- **Hybrid retrieval** combines BM25 keyword search with semantic vector search and Reciprocal Rank Fusion. If the embedding model cannot load, keyword search remains available.
- **PDF / DOCX / TXT / Markdown ingestion**, page-aware PDF citations, table extraction where supported, and overlap-aware chunking.
- **Grounded answers and abstention**: the relevance step can refuse questions not supported by the uploaded material.
- **Answer verification**: a verifier checks drafted claims against retrieved excerpts and can request a revision.
- **Source excerpts**, chat history for the current session, adjustable retrieval settings, and a lightweight evaluation script.
- **Offline smoke tests** for core parsing, retrieval, and pipeline behavior (no Ollama model required).

## Architecture

```text
Upload files
   ↓
Text + table extraction → page-aware chunks
   ↓
FastEmbed vectors → persistent ChromaDB
   ↘ BM25 keyword index
       ↓
Reciprocal Rank Fusion
       ↓
Relevance check → research answer with [1] citations
       ↓
Verification → answer + source excerpts
```

> Note: the BM25 index and chat session are in memory. Chroma's vector collection persists locally. Re-upload the same documents to rebuild the in-memory keyword index when restarting the app.

## Requirements

- Python 3.10–3.12 recommended
- Ollama installed and running
- 8 GB RAM recommended (small models can run on less, more slowly)
- Internet connection for the initial Ollama and embedding-model downloads

## Setup on Windows

1. Install Ollama from [ollama.com/download](https://ollama.com/download), then open it.
2. Pull a model that fits your machine:

```powershell
ollama pull qwen2.5:3b
# Smaller option for lower memory:
ollama pull qwen2.5:1.5b
```

3. Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the offline self-check:

```powershell
python smoke_test.py
```

5. Launch the app:

```powershell
streamlit run app.py
```

The first indexing run downloads the FastEmbed embedding model. Later runs reuse the local cache. Ollama models are downloaded separately using `ollama pull`.

## Setup on macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python smoke_test.py
streamlit run app.py
```

## How to use

1. Start Ollama and confirm a model is installed (`ollama list`).
2. Launch Streamlit and upload one or more documents in the sidebar.
3. Wait for indexing to finish.
4. Ask specific questions such as “What was the reported year-over-year change in operating expenses?” or “Compare the risk factors discussed in sections 2 and 4.”
5. Expand **Sources** under the answer to inspect retrieved excerpts and page numbers. Treat a warning or inconclusive verification as a signal to check the source before relying on the answer.

## Evaluate on your own reports

Create a JSON file based on `eval_questions.example.json`, with questions you can verify from your report. Example:

```json
[
  {
    "question": "What was the reported operating margin?",
    "must_contain": ["18.4%"],
    "must_not_contain": ["17.9%"]
  },
  {
    "question": "What was the company's office temperature in 1982?",
    "out_of_scope": true
  }
]
```

Run:

```powershell
python evaluate.py --docs reports\annual-report.pdf --questions eval_questions.json --model qwen2.5:3b
```

The evaluation script reports basic expected-string and out-of-scope checks; it is a practical regression aid, not a formal measure of factual accuracy.

## Project layout

```text
docchat-local-rag/
├── app.py
├── evaluate.py
├── smoke_test.py
├── requirements.txt
├── docchat/
│   ├── __init__.py
│   ├── agents.py
│   ├── llm.py
│   ├── loader.py
│   └── retriever.py
├── data/
│   └── chroma/          # created at runtime; local vector database
└── .cache/               # created at runtime; embedding model cache
```

