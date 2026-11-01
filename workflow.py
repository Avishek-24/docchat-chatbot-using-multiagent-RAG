from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from agents.relevance_checker import RelevanceChecker
from agents.research_agent import ResearchAgent
from agents.verification_agent import VerificationAgent

MAX_ATTEMPTS = 3   # 1 first try + up to 2 self-correction retries


class AgentState(TypedDict):
    question: str
    retriever: object
    docs: List[Document]
    relevance: str
    draft_answer: str
    context_used: str
    verification_report: Dict
    final_output: str
    # new fields for the agentic loop
    attempts: int
    feedback: str
    search_queries: List[str]
    passed: bool


class AgentWorkflow:
    def __init__(self):
        self.relevance_checker = RelevanceChecker()
        self.research_agent = ResearchAgent()
        self.verification_agent = VerificationAgent()
        self.app = self._build_graph()

    # ---------- nodes ----------
    def _check_relevance_step(self, state: AgentState) -> AgentState:
        rel = self.relevance_checker.check(state["question"], state["retriever"])
        return {**state, "relevance": rel}

    def _research_step(self, state: AgentState) -> AgentState:
        # The LLM now decides what to search; feedback is empty on the first try
        res = self.research_agent.generate_draft(
            state["question"], state["retriever"], feedback=state["feedback"] or None
        )
        return {
            **state,
            "draft_answer": res["draft_answer"],
            "context_used": res["context_used"],
            "docs": res["docs"],
            "search_queries": state["search_queries"] + res["search_queries"],
        }

    def _verification_step(self, state: AgentState) -> AgentState:
        v_res = self.verification_agent.verify(state["draft_answer"], state["context_used"])
        ok = self.verification_agent.passed(v_res)
        attempts = state["attempts"] + 1
        feedback = "" if ok else self.verification_agent.build_feedback(v_res)

        report = self.verification_agent.format_report(v_res)
        note = ""
        if not ok:
            note = "\n\n⚠️ _Could not be fully verified after retries; treat with caution._"
        final_answer = (
            f"{state['draft_answer']}\n\n---\n### 🔍 Verification Report\n{report}"
            f"\n\n_Attempts: {attempts}_{note}"
        )
        return {
            **state,
            "verification_report": v_res,
            "final_output": final_answer,
            "attempts": attempts,
            "feedback": feedback,
            "passed": ok,
        }

    # ---------- routing ----------
    def _decide_after_relevance(self, state: AgentState) -> str:
        return "end" if state["relevance"] == "NO_MATCH" else "research"

    def _decide_after_verify(self, state: AgentState) -> str:
        if state["passed"] or state["attempts"] >= MAX_ATTEMPTS:
            return "end"
        return "retry"      # self-correction loop

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)

        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance,
            {"research": "research", "end": END},
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._decide_after_verify,
            {"retry": "research", "end": END},
        )
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
            "final_output": "",
            "attempts": 0,
            "feedback": "",
            "search_queries": [],
            "passed": False,
        }
        return self.app.invoke(initial_state)
