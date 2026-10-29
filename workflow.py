from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from agents.relevance_checker import RelevanceChecker
from agents.research_agent import ResearchAgent
from agents.verification_agent import VerificationAgent

class AgentState(TypedDict):
    question: str
    retriever: object
    docs: List[Document]
    relevance: str
    draft_answer: str
    context_used: str
    verification_report: Dict
    final_output: str

class AgentWorkflow:
    def __init__(self):
        self.relevance_checker = RelevanceChecker()
        self.research_agent = ResearchAgent()
        self.verification_agent = VerificationAgent()
        self.app = self._build_graph()

    def _check_relevance_step(self, state: AgentState) -> AgentState:
        rel = self.relevance_checker.check(state["question"], state["retriever"])
        docs = state["retriever"].invoke(state["question"]) if rel != "NO_MATCH" else []
        return {**state, "relevance": rel, "docs": docs}

    def _research_step(self, state: AgentState) -> AgentState:
        res = self.research_agent.generate_draft(state["question"], state["docs"])
        return {**state, "draft_answer": res["draft_answer"], "context_used": res["context_used"]}

    def _verification_step(self, state: AgentState) -> AgentState:
        v_res = self.verification_agent.verify(state["draft_answer"], state["context_used"])
        report = self.verification_agent.format_report(v_res)
        final_answer = f"{state['draft_answer']}\n\n---\n### 🔍 Verification Report\n{report}"
        return {**state, "verification_report": v_res, "final_output": final_answer}

    def _decide_after_relevance(self, state: AgentState) -> str:
        if state["relevance"] == "NO_MATCH":
            return "end"
        return "research"

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)

        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance,
            {"research": "research", "end": END}
        )
        workflow.add_edge("research", "verify")
        workflow.add_edge("verify", END)

        return workflow.compile()

    def run(self, question: str, retriever) -> Dict:
        initial_state = {
            "question": question,
            "retriever": retriever,
            "docs": [],
            "relevance": "",
            "draft_answer": "",
            "context_used": "",
            "verification_report": {},
            "final_output": ""
        }
        return self.app.invoke(initial_state)