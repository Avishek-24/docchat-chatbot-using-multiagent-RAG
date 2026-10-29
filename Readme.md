# 📄 DocChat: Multi-Agent Local RAG Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LangGraph%20%7C%20Streamlit-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Parser](https://img.shields.io/badge/Parser-Docling-green.svg)](https://github.com/DS4SD/docling)
[![LLM](https://img.shields.io/badge/Ollama-Llama--3.2-blueviolet.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DocChat** is an enterprise-grade, privacy-centric Retrieval-Augmented Generation (RAG) platform designed to parse, index, search, and synthesize information from dense, structured corporate documents (such as central bank regulatory reports, quarterly financial filings, and policy whitepapers) with **zero hallucination tolerance**.

Unlike standard RAG pipelines that fail on complex tables or hallucinate unmentioned facts, DocChat combines **Docling** visual document layout analysis, **Hybrid Retrieval (BM25 + Vector Search via Reciprocal Rank Fusion)**, and a **3-Tier Multi-Agent System orchestrated with LangGraph**.

---

## 🌟 Key Features

* **100% Local & Privacy-Compliant:** Runs completely on-premises using [Ollama](https://ollama.com/) (`llama3.2` for reasoning, `mxbai-embed-large` for dense embeddings). No API keys or cloud dependencies required.
* **Layout-Aware Parsing (Docling):** Preserves multi-column text, footnotes, and complex financial tables by parsing documents directly into structured Markdown format.
* **Hybrid Search with RRF:** Combines BM25 keyword matching (for exact figures, acronyms, and codes) with ChromaDB dense vector search, merged seamlessly using Reciprocal Rank Fusion (RRF).
* **Multi-Agent Verification Pipeline (LangGraph):**
  1. **Relevance Checker Agent:** Verifies if the user's question can be answered from the retrieved context. Immediately halts execution on out-of-scope queries (`NO_MATCH`).
  2. **Research Agent:** Synthesizes factual draft answers grounded strictly in verified context.
  3. **Verification Agent:** Fact-checks draft answers against raw source text, generating an automated verification report (`Supported`, `Unsupported Claims`, `Contradictions`).
* **Intelligent Content Caching:** Built-in SHA-256 byte hashing and pickle store caching to prevent redundant file re-parsing.
* **High-Performance UI:** Built with Streamlit, optimized with `@st.cache_resource` for sub-second startup times and smooth state management.

---

## 📐 System Architecture

```
                                 ┌──────────────────────────┐
                                 │   Uploaded Document(s)   │
                                 └────────────┬─────────────┘
                                              │
                                   Docling Markdown Parsing
                                   & SHA-256 Pickle Caching
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │ Markdown Header Chunking │
                                 └────────────┬─────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         ┌─────────────────────┐                             ┌─────────────────────┐
         │    BM25 Search      │                             │   ChromaDB Vector   │
         │  (Lexical Matching) │                             │   (mxbai-embed)     │
         └──────────┬──────────┘                             └──────────┬──────────┘
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                               Reciprocal Rank Fusion (RRF)
                                              │
                                              ▼
                                ┌───────────────────────────┐
                                │   Agent 1: Relevance      │
                                │   Checker (Llama 3.2)     │
                                └─────────────┬─────────────┘
                                              │
                                  Is Query in Context?
                                   /                 \
                          [NO_MATCH]                 [CAN_ANSWER]
                             /                         \
                            ▼                           ▼
                 ┌────────────────────┐     ┌───────────────────────┐
                 │ Halt Pipeline      │     │  Agent 2: Research    │
                 │ (Out-of-Scope Alert)│     │  Agent (Draft Answer) │
                 └────────────────────┘     └───────────┬───────────┘
                                                        │
                                                        ▼
                                            ┌───────────────────────┐
                                            │ Agent 3: Verification │
                                            │ Agent (Fact Checking) │
                                            └───────────┬───────────┘
                                                        │
                                                        ▼
                                            ┌───────────────────────┐
                                            │ Verified Final Answer │
                                            │ + Fact-Check Report   │
                                            └───────────────────────┘
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.10 or higher
* [Ollama](https://ollama.com/) installed and running on your system

### 2. Install & Start Ollama Models
Open your terminal and pull the required open-weights models:

```bash
# Start Ollama service (if not running)
ollama serve

# Pull the primary LLM (Llama 3.2 3B)
ollama pull llama3.2

# Pull the embedding model (mxbai-embed-large)
ollama pull mxbai-embed-large
```

### 3. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/DocChat.git
cd DocChat

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

1. **Upload Documents:** Use the sidebar to upload financial reports, PDFs, or Markdown files. Click **Process Documents**.
2. **Ask Questions:** Enter your question in the query input box and hit **Submit Query**.
3. **Review Results:** View the generated answer along with the real-time **Verification Report**.

---

## 📁 Repository Structure

```
DocChat/
├── agents/
│   ├── relevance_checker.py   # Agent 1: Context scope validation
│   ├── research_agent.py      # Agent 2: Grounded answer drafting
│   └── verification_agent.py  # Agent 3: Fact-checking critic
├── config/
│   ├── constants.py           # File size limits and global constants
│   └── settings.py            # Cache directory & timeout settings
├── .cache/                    # Automated SHA-256 pickle cache directory
├── app.py                     # Streamlit frontend application
├── builder.py                 # Hybrid retriever (BM25 + Vector Search via RRF)
├── file_handler.py            # Docling parsing & chunking engine
├── requirements.txt           # Project dependencies
├── workflow.py                # LangGraph state graph orchestrator
└── README.md                  # Project documentation
```

---





