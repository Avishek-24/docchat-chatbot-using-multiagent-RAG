from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

class RelevanceChecker:
    def __init__(self, model_name: str = "llama3.2"):
        self.llm = ChatOllama(model=model_name, temperature=0.0)

    def check(self, question: str, retriever, k: int = 8) -> str:
        docs = retriever.invoke(question)
        if not docs:
            return "NO_MATCH"

        context = "\n\n".join([doc.page_content for doc in docs[:k]])
        prompt = f"""You are an AI relevance checker between a user's question and provided document excerpts.

Instructions:
Classify how well the document context addresses the user's question.
Respond with ONLY ONE of the following labels: CAN_ANSWER, PARTIAL, or NO_MATCH. Do not add any explanation or preamble.

Question: {question}

Passages:
{context}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        llm_response = response.content.strip().upper()

        if llm_response in ["CAN_ANSWER", "PARTIAL", "NO_MATCH"]:
            return llm_response
        return "NO_MATCH"