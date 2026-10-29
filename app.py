import os
import tempfile
import streamlit as st

from file_handler import DocumentProcessor
from builder import RetrieverBuilder
from workflow import AgentWorkflow

st.set_page_config(page_title="DocChat - Multi-Agent RAG", page_icon="📄", layout="wide")

st.title("📄 DocChat: Multi-Agent Local RAG Assistant")
st.markdown("Powered by **Streamlit**, **Ollama (Llama 3.2)**, **Docling**, and **LangGraph**")

# Session state initialization
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
if "builder" not in st.session_state:
    st.session_state.builder = RetrieverBuilder()
if "workflow" not in st.session_state:
    st.session_state.workflow = AgentWorkflow()
if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary") and uploaded_files:
        with st.spinner("Parsing & embedding documents..."):
            temp_paths = []
            temp_dir = tempfile.mkdtemp()
            for u_file in uploaded_files:
                p = os.path.join(temp_dir, u_file.name)
                with open(p, "wb") as f:
                    f.write(u_file.getbuffer())
                temp_paths.append(p)

            chunks = st.session_state.processor.process(temp_paths)
            st.session_state.retriever = st.session_state.builder.build_hybrid_retriever(chunks)
            st.success(f"Successfully processed {len(chunks)} chunks!")

st.header("2. Ask Questions")
query = st.text_input("Enter your question based on uploaded documents:")

if st.button("Submit Query") and query:
    if not st.session_state.retriever:
        st.error("Please upload and process documents first!")
    else:
        with st.spinner("Running Multi-Agent Verification Pipeline..."):
            result = st.session_state.workflow.run(query, st.session_state.retriever)

            if result["relevance"] == "NO_MATCH":
                st.warning("⚠️ The query is out of scope or not mentioned in the uploaded documents.")
            else:
                st.markdown("### Answer")
                st.markdown(result["final_output"])