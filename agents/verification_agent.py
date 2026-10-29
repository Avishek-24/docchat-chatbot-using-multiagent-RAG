from typing import Dict, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

class VerificationAgent:
    def __init__(self, model_name: str = "llama3.2"):
        self.llm = ChatOllama(model=model_name, temperature=0.0)

    def verify(self, answer: str, context: str) -> Dict:
        prompt = f"""You are an AI verification assistant. Verify the accuracy and relevance of the answer against the context.

Verify the following:
1. Supported: YES or NO
2. Unsupported Claims: None or comma-separated list
3. Contradictions: None or comma-separated list
4. Relevant: YES or NO

Respond strictly in the exact format below:
Supported: YES/NO
Unsupported Claims: None
Contradictions: None
Relevant: YES/NO

Answer: {answer}

Context:
{context}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._parse_response(response.content.strip())

    def _parse_response(self, text: str) -> Dict:
        verification = {
            "Supported": "NO",
            "Unsupported Claims": "None",
            "Contradictions": "None",
            "Relevant": "NO"
        }
        for line in text.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if k in verification:
                    verification[k] = v
        return verification

    def format_report(self, verification: Dict) -> str:
        return (
            f"**Supported:** {verification.get('Supported', 'NO')}\n"
            f"**Unsupported Claims:** {verification.get('Unsupported Claims', 'None')}\n"
            f"**Contradictions:** {verification.get('Contradictions', 'None')}\n"
            f"**Relevant:** {verification.get('Relevant', 'NO')}"
        )