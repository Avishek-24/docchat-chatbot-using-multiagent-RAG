from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

class ResearchAgent:
    def __init__(self, model_name: str = "llama3.2"):
        self.llm = ChatOllama(model=model_name, temperature=0.3)

    def generate_draft(self, question: str, documents: List[Document]) -> Dict:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"""You are an AI assistant designed to provide precise and factual answers based on provided context.

Instructions:
- Answer the question using ONLY the provided context.
- Be clear, concise, and factual.
- If the context lacks sufficient details, state what is missing.

Question: {question}

Context:
{context}

Provide your answer below:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        draft_answer = response.content.strip()

        if not draft_answer:
            draft_answer = "I cannot answer this question based on the provided documents."

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }