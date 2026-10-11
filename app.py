
from __future__ import annotations

import hashlib

import streamlit as st

from docchat.agents import run_pipeline
from docchat.llm import DEFAULT_HOST, LLMError, OllamaClient
from docchat.loader import ALLOWED_TYPES, build_chunks
from docchat.retriever import Embedder, HybridRetriever

st.set_page_config(page_title="DocChat Lite", page_icon="📄", layout="wide")

PREFERRED_MODELS = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "qwen2.5:1.5b", "llama3.2:1b"]

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("retriever", None)
ss.setdefault("index_key", None)
ss.setdefault("index_warnings", [])
ss.setdefault("index_note", "")



@st.cache_resource(show_spinner="Loading the embedding model (first run downloads ~130 MB)…")
def load_embedder():
    try:
        return Embedder(), None
    except Exception as exc:  
        return None, str(exc)



def render_extras(msg: dict) -> None:
    """Verification badge + sources, shown under an assistant answer."""
    v = msg.get("verification")
    if msg.get("relevance") == "NO_MATCH":
        st.info("Out of scope: nothing in your documents matched this question.")
    elif v:
        if v["verdict"] == "verified":
            st.success("✅ Verified: the claims in this answer were found in the source excerpts.")
        elif v["verdict"] == "issues":
            st.warning("⚠️ Some claims could not be confirmed in the documents. "
                       "Check the sources below before relying on this answer.")
        else:
            st.info("ℹ️ Verification was inconclusive. Check the sources below.")
        if msg.get("revisions"):
            st.caption(f"The answer was revised {msg['revisions']}× after the fact-check.")
        details = v["unsupported_claims"] + v["contradictions"]
        if details or v.get("notes"):
            with st.expander("Verification report"):
                if v["unsupported_claims"]:
                    st.markdown("**Unsupported claims**")
                    for c in v["unsupported_claims"]:
                        st.markdown(f"- {c}")
                if v["contradictions"]:
                    st.markdown("**Contradictions**")
                    for c in v["contradictions"]:
                        st.markdown(f"- {c}")
                if v.get("notes"):
                    st.caption(v["notes"])

    sources = msg.get("sources") or []
    if sources:
        with st.expander(f"📚 Sources ({len(sources)})"):
            for n, s in enumerate(sources, start=1):
                page = f", page {s['page']}" if s.get("page") else ""
                st.markdown(f"**[{n}] {s['source']}{page}**" + (" · table" if s["kind"] == "table" else ""))
                st.text(s["text"][:700])


def to_message(result) -> dict:
    return {
        "role": "assistant",
        "content": result.answer,
        "relevance": result.relevance,
        "verification": result.verification,
        "revisions": result.revisions,
        "sources": [
            {"source": h.chunk.source, "page": h.chunk.page, "kind": h.chunk.kind, "text": h.chunk.text}
            for h in result.hits
        ],
    }


def answer_live(question: str, client: OllamaClient, k: int, verify: bool, retries: int) -> dict:
    status_box, answer_box = st.empty(), st.empty()
    text, result = "", None
    try:
        for ev in run_pipeline(question, ss.retriever, client, k=k, verify=verify, max_retries=retries):
            if ev.kind == "status":
                status_box.caption(ev.payload)
            elif ev.kind == "reset":
                text = ""
                answer_box.empty()
            elif ev.kind == "token":
                text += ev.payload
                answer_box.markdown(text + "▌")
            elif ev.kind == "done":
                result = ev.payload
    except LLMError as exc:
        status_box.empty()
        answer_box.empty()
        st.error(str(exc))
        return {"role": "assistant", "content": f"⚠️ {exc}"}
    except Exception as exc:  # never show a traceback to the user
        status_box.empty()
        answer_box.empty()
        st.error(f"Something went wrong while answering: {exc}")
        return {"role": "assistant", "content": f"⚠️ Something went wrong: {exc}"}

    status_box.empty()
    msg = to_message(result)
    answer_box.markdown(msg["content"])
    render_extras(msg)
    return msg



with st.sidebar:
    st.title("📄 DocChat Lite")
    st.caption("Ask questions about your documents. Runs on your computer, free.")

    with st.expander("Advanced connection"):
        host = st.text_input("Ollama address", DEFAULT_HOST)
    client = OllamaClient(host=host)
    models = client.list_models()
    model = None

    if models is None:
        st.error("**Ollama isn't running.**\n\n"
                 "1. Install it from ollama.com\n"
                 "2. Open the Ollama app (or run `ollama serve`)\n"
                 "3. Run `ollama pull qwen2.5:3b`")
        st.button("🔄 Check again")
    elif not models:
        st.warning("Ollama is running but has no models yet.\n\nRun `ollama pull qwen2.5:3b` in a terminal, then click below.")
        st.button("🔄 Check again")
    else:
        default = next((i for i, m in enumerate(models) if m in PREFERRED_MODELS), 0)
        model = st.selectbox("Language model", models, index=default)
        client.model = model

    st.divider()
    uploaded = st.file_uploader("Upload documents", type=ALLOWED_TYPES, accept_multiple_files=True,
                                help="PDF, Word (.docx), text or Markdown")

    with st.expander("Settings"):
        k = st.slider("Passages to retrieve", 3, 8, 6)
        verify = st.checkbox("Fact-check answers (slower, safer)", value=True)
        retries = st.slider("Max revisions after fact-check", 0, 2, 1, disabled=not verify)
        use_tables = st.checkbox("Extract tables from PDFs (slower)", value=True)

    if ss.messages and st.button("🗑️ Clear chat"):
        ss.messages = []
        st.rerun()

# --------------------------------------------------------------------------
# Build the index when the uploaded files change
# --------------------------------------------------------------------------
files = [(f.name, f.getvalue()) for f in (uploaded or [])]
key = (tuple(sorted((n, hashlib.sha256(b).hexdigest()) for n, b in files)), use_tables) if files else None

if key != ss.index_key:
    ss.index_key, ss.messages, ss.retriever = key, [], None
    ss.index_warnings, ss.index_note = [], ""
    if files:
        with st.sidebar.status("Reading your documents…", expanded=True) as status:
            chunks, warnings = build_chunks(
                files, extract_tables=use_tables, on_file=lambda n: st.write(f"📖 {n}"))
            ss.index_warnings = warnings
            if not chunks:
                status.update(label="No readable text found", state="error")
            else:
                embedder, embed_error = load_embedder()
                bar = st.progress(0.0, text="Indexing…")
                retriever = HybridRetriever(
                    chunks, embedder,
                    progress=lambda f: bar.progress(min(f, 1.0), text=f"Indexing… {int(f * 100)}%"))
                bar.empty()
                ss.retriever = retriever
                ss.index_note = f"{len(chunks)} passages from {len(files)} file(s) · search mode: {retriever.mode}"
                if embedder is None or retriever.dense_error:
                    ss.index_warnings.append(
                        "The embedding model couldn't load, so search is keyword-only. "
                        "Connect to the internet once so it can download, then restart the app.")
                status.update(label="Ready", state="complete", expanded=False)

for w in ss.index_warnings:
    st.sidebar.warning(w)
if ss.index_note:
    st.sidebar.caption(ss.index_note)

# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
st.header("Chat with your documents")

if ss.retriever is None:
    st.info("👈 Upload at least one document in the sidebar to get started.")
elif model is None:
    st.warning("Start Ollama and pick a model in the sidebar to begin chatting.")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_extras(m)

question = st.chat_input("Ask a question about your documents…",
                         disabled=(ss.retriever is None or model is None))
if question:
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        ss.messages.append(answer_live(question, client, k, verify, retries))
