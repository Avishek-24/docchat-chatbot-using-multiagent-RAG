from typing import List, Dict, Optional
from langchain_groq import ChatGroq
#from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.tools import tool


SYSTEM_PROMPT = """You are a research assistant that answers questions about the user's uploaded documents.

You have a tool called `search_documents`. Rules:
- ALWAYS call it before answering. You decide what to search for.
- If the question has several parts, search for each part separately.
- If results look incomplete, search again with different keywords.
- When you have enough information, stop calling tools and write the final answer.
- Answer using ONLY the retrieved passages. If something is missing, say what is missing.
- Be clear, concise, and factual."""


class ResearchAgent:
    """Tool-calling agent: the LLM decides when and what to search."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b", max_steps: int = 4):
        self.llm = ChatGroq(model=model_name, temperature=0.3)
        #self.llm = ChatOllama(model=model_name, temperature=0.3)
        self.max_steps = max_steps

    # ---------- tool factory ----------
    def _make_search_tool(self, retriever, collected: List[Document]):
        @tool
        def search_documents(query: str) -> str:
            """Search the uploaded documents and return the most relevant passages.
            Use short, specific keyword queries."""
            docs = retriever.invoke(query)
            collected.extend(docs)
            if not docs:
                return "No results found. Try different keywords."
            return "\n\n".join(
                f"[{i + 1}] (source: {d.metadata.get('source', 'unknown')})\n{d.page_content}"
                for i, d in enumerate(docs)
            )
        return search_documents

    @staticmethod
    def _dedupe(docs: List[Document]) -> List[Document]:
        seen, unique = set(), []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                unique.append(d)
        return unique

    # ---------- main entry ----------
    def generate_draft(self, question: str, retriever, feedback: Optional[str] = None) -> Dict:
        collected: List[Document] = []
        queries: List[str] = []
        search_tool = self._make_search_tool(retriever, collected)
        llm_with_tools = self.llm.bind_tools([search_tool])

        user_msg = f"Question: {question}"
        if feedback:
            user_msg += (
                "\n\nYour previous answer was rejected by a fact-checker for these reasons:\n"
                f"{feedback}\n"
                "Search again, fix these problems, and only state what the passages support."
            )

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)]
        draft_answer = ""

        for _ in range(self.max_steps):
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:          # model decided it is done
                draft_answer = ai_msg.content.strip()
                break

            for call in ai_msg.tool_calls:     # model decided to search
                result = search_tool.invoke(call["args"])
                queries.append(call["args"].get("query", ""))
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        else:
            # step budget used up: force a final answer without tools
            messages.append(HumanMessage(
                content="Stop searching. Write the final answer now using only the passages above."))
            draft_answer = self.llm.invoke(messages).content.strip()

        # Safety net: small models sometimes skip the tool entirely
        if not collected:
            fallback_docs = retriever.invoke(question)
            collected.extend(fallback_docs)
            queries.append(f"(fallback) {question}")
            ctx = "\n\n".join(d.page_content for d in fallback_docs)
            prompt = (
                "Answer using ONLY the context below. If details are missing, say what is missing.\n\n"
                f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"
            )
            draft_answer = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()

        docs = self._dedupe(collected)
        context = "\n\n".join(d.page_content for d in docs)

        if not draft_answer:
            draft_answer = "I cannot answer this question based on the provided documents."

        return {
            "draft_answer": draft_answer,
            "context_used": context,
            "docs": docs,
            "search_queries": queries,
        }
