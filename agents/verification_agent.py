from typing import Dict
from langchain_groq import ChatGroq
#from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


class VerificationAgent:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(model=model_name, temperature=0.0)
        #self.llm = ChatOllama(model=model_name, temperature=0.0)

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
            "Relevant": "NO",
        }
        for line in text.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip().strip("*").strip(), v.strip().strip("*").strip()
                if k in verification:
                    verification[k] = v
        return verification

    @staticmethod
    def _is_none(value: str) -> bool:
        return value.strip().lower().rstrip(".") in ("", "none", "n/a", "no")

    def passed(self, v: Dict) -> bool:
        """Answer is accepted only if supported, relevant and free of problems."""
        return (
            v.get("Supported", "NO").upper().startswith("YES")
            and v.get("Relevant", "NO").upper().startswith("YES")
            and self._is_none(v.get("Unsupported Claims", "None"))
            and self._is_none(v.get("Contradictions", "None"))
        )

    def build_feedback(self, v: Dict) -> str:
        """Turn a failed verification into instructions for the research agent."""
        parts = []
        if not v.get("Supported", "NO").upper().startswith("YES"):
            parts.append("- The answer is not fully supported by the documents.")
        if not self._is_none(v.get("Unsupported Claims", "None")):
            parts.append(f"- Unsupported claims: {v['Unsupported Claims']}")
        if not self._is_none(v.get("Contradictions", "None")):
            parts.append(f"- Contradictions with the documents: {v['Contradictions']}")
        if not v.get("Relevant", "NO").upper().startswith("YES"):
            parts.append("- The answer does not address the question.")
        return "\n".join(parts)

    def format_report(self, verification: Dict) -> str:
        return (
            f"**Supported:** {verification.get('Supported', 'NO')}\n"
            f"**Unsupported Claims:** {verification.get('Unsupported Claims', 'None')}\n"
            f"**Contradictions:** {verification.get('Contradictions', 'None')}\n"
            f"**Relevant:** {verification.get('Relevant', 'NO')}"
        )
