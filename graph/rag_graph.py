"""
graph/rag_graph.py

This is where all agents connect into a pipeline using LangGraph.

LangGraph thinks in nodes and edges:
- Nodes = agents (functions that take state, return state)
- Edges = connections between agents
- Conditional edges = routing logic (if critic fails → repair)

Visual flow:
retriever → critic → [if pass] → answer_generator → verifier → END
                   → [if fail] → repair → retriever (loop)
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from graph.state import RAGState, HealingStatus, VerificationStatus
from agents.retriever_agent import retriever_agent
from agents.critic_agent import critic_agent
from agents.repair_agent import repair_agent
from agents.verifier_agent import verifier_agent

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
MAX_LOOPS = int(os.getenv("MAX_HEALING_LOOPS", 3))

# ── Answer Generator (inline node, not a separate agent file) ─────────────────
ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions based on retrieved documents.

Question: {query}

Retrieved Context:
{context}

Instructions:
- Answer ONLY based on the provided context
- If context doesn't fully answer the question, say so honestly
- Be concise and clear
- Cite sources where possible (e.g. "According to filename.pdf page 3...")

Answer:
""")

def answer_generator(state: RAGState) -> RAGState:
    """Generate final answer from retrieved chunks."""
    state.log("[Generator] Generating answer from retrieved chunks...")

    if not state.retrieved_chunks:
        state.final_answer = "I could not find relevant information to answer your question."
        state.log("[Generator] No chunks available, returning default answer.")
        return state

    context = "\n\n".join([
        f"[{c['source']} | Page {c['page']}]\n{c['text']}"
        for c in state.retrieved_chunks[:5]
    ])

    try:
        llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        chain = ANSWER_PROMPT | llm

        response = chain.invoke({
            "query": state.query,
            "context": context
        })

        state.final_answer = response.content.strip()
        state.answer_sources = [
            {"source": c["source"], "page": c["page"], "score": c["score"]}
            for c in state.retrieved_chunks[:5]
        ]
        state.log(f"[Generator] Answer generated ({len(state.final_answer)} chars)")

    except Exception as e:
        state.error = str(e)
        state.final_answer = f"Error generating answer: {e}"
        state.log(f"[Generator] ERROR: {e}")

    return state


# ── Routing Logic ─────────────────────────────────────────────────────────────
def route_after_critic(state: RAGState) -> str:
    """
    This is the self-healing decision point.

    Called after critic_agent runs.
    Returns the name of the NEXT node to go to.

    Logic:
    - Critic passed → generate answer
    - Critic failed + loops remaining → repair and retry
    - Critic failed + no loops left → generate anyway (best effort)
    """
    if state.critic_passed:
        state.log("[Router] Critic passed → generating answer")
        return "answer_generator"

    if state.healing_loop_count >= MAX_LOOPS:
        state.log("[Router] Max loops reached → generating best effort answer")
        return "answer_generator"

    state.log(f"[Router] Critic failed → triggering repair (loop {state.healing_loop_count + 1})")
    return "repair_agent"


def route_after_verification(state: RAGState) -> str:
    """
    After verification:
    - Verified → END (send answer to user)
    - Hallucinated → END (but answer will include warning)
    """
    if state.verification_status == VerificationStatus.HALLUCINATED:
        state.final_answer = (
            f"⚠️ Warning: This answer may contain unverified information.\n\n"
            f"{state.final_answer}"
        )
        state.log("[Router] Hallucination detected — adding warning to answer")
    return END


# ── Build the Graph ───────────────────────────────────────────────────────────
def build_rag_graph():
    """
    Assemble all nodes and edges into a LangGraph pipeline.
    Returns a compiled graph ready to run.
    """

    # RAGState drives everything — LangGraph passes it between nodes
    # We use a dict-based state for LangGraph compatibility
    from typing import TypedDict, List, Dict, Any, Optional

    class GraphState(TypedDict):
        query: str
        rephrased_query: str
        retrieved_chunks: List[Dict[str, Any]]
        critic_score: float
        critic_feedback: str
        critic_passed: bool
        healing_status: str
        healing_loop_count: int
        max_healing_loops: int
        final_answer: str
        answer_sources: List[Dict[str, Any]]
        verification_status: str
        verification_feedback: str
        error: Optional[str]
        run_log: List[str]

    def retriever_node(state: dict) -> dict:
        rag_state = dict_to_ragstate(state)
        result = retriever_agent(rag_state)
        return ragstate_to_dict(result)

    def critic_node(state: dict) -> dict:
        rag_state = dict_to_ragstate(state)
        result = critic_agent(rag_state)
        return ragstate_to_dict(result)

    def repair_node(state: dict) -> dict:
        rag_state = dict_to_ragstate(state)
        result = repair_agent(rag_state)
        return ragstate_to_dict(result)

    def answer_node(state: dict) -> dict:
        rag_state = dict_to_ragstate(state)
        result = answer_generator(rag_state)
        return ragstate_to_dict(result)

    def verifier_node(state: dict) -> dict:
        rag_state = dict_to_ragstate(state)
        result = verifier_agent(rag_state)
        return ragstate_to_dict(result)

    def route_critic(state: dict) -> str:
        rag_state = dict_to_ragstate(state)
        return route_after_critic(rag_state)

    def route_verifier(state: dict) -> str:
        rag_state = dict_to_ragstate(state)
        return route_after_verification(rag_state)

    # Build graph
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("retriever_agent", retriever_node)
    graph.add_node("critic_agent", critic_node)
    graph.add_node("repair_agent", repair_node)
    graph.add_node("answer_generator", answer_node)
    graph.add_node("verifier_agent", verifier_node)

    # Add edges
    graph.set_entry_point("retriever_agent")
    graph.add_edge("retriever_agent", "critic_agent")

    # Conditional edge — self-healing decision point
    graph.add_conditional_edges(
        "critic_agent",
        route_critic,
        {
            "answer_generator": "answer_generator",
            "repair_agent": "repair_agent"
        }
    )

    # Repair loops back to retriever
    graph.add_edge("repair_agent", "retriever_agent")

    # Answer goes to verifier
    graph.add_edge("answer_generator", "verifier_agent")

    # Verifier ends the pipeline
    graph.add_conditional_edges(
        "verifier_agent",
        route_verifier,
        {END: END}
    )

    return graph.compile()


# ── State conversion helpers ───────────────────────────────────────────────────
def dict_to_ragstate(d: dict) -> RAGState:
    """Convert LangGraph dict state → RAGState dataclass."""
    state = RAGState()
    state.query = d.get("query", "")
    state.rephrased_query = d.get("rephrased_query", "")
    state.retrieved_chunks = d.get("retrieved_chunks", [])
    state.critic_score = d.get("critic_score", 0.0)
    state.critic_feedback = d.get("critic_feedback", "")
    state.critic_passed = d.get("critic_passed", False)
    state.healing_loop_count = d.get("healing_loop_count", 0)
    state.max_healing_loops = d.get("max_healing_loops", 3)
    state.final_answer = d.get("final_answer", "")
    state.answer_sources = d.get("answer_sources", [])
    state.error = d.get("error", None)
    state.run_log = d.get("run_log", [])

    # Convert string enums back
    try:
        state.healing_status = HealingStatus(d.get("healing_status", "not_started"))
    except ValueError:
        state.healing_status = HealingStatus.NOT_STARTED

    try:
        state.verification_status = VerificationStatus(d.get("verification_status", "not_verified"))
    except ValueError:
        state.verification_status = VerificationStatus.NOT_VERIFIED

    state.verification_feedback = d.get("verification_feedback", "")
    return state


def ragstate_to_dict(state: RAGState) -> dict:
    """Convert RAGState dataclass → LangGraph dict state."""
    return {
        "query": state.query,
        "rephrased_query": state.rephrased_query,
        "retrieved_chunks": state.retrieved_chunks,
        "critic_score": state.critic_score,
        "critic_feedback": state.critic_feedback,
        "critic_passed": state.critic_passed,
        "healing_status": state.healing_status.value,
        "healing_loop_count": state.healing_loop_count,
        "max_healing_loops": state.max_healing_loops,
        "final_answer": state.final_answer,
        "answer_sources": state.answer_sources,
        "verification_status": state.verification_status.value,
        "verification_feedback": state.verification_feedback,
        "error": state.error,
        "run_log": state.run_log
    }


# ── Run the pipeline ──────────────────────────────────────────────────────────
def run_pipeline(query: str, initial_state: dict = None) -> dict:
    """
    Main entry point. Call this to run the full pipeline.

    Usage:
        result = run_pipeline("What is LangGraph?")
        print(result["final_answer"])
    """
    graph = build_rag_graph()

    state = initial_state or {
        "query": query,
        "rephrased_query": "",
        "retrieved_chunks": [],
        "critic_score": 0.0,
        "critic_feedback": "",
        "critic_passed": False,
        "healing_status": "not_started",
        "healing_loop_count": 0,
        "max_healing_loops": MAX_LOOPS,
        "final_answer": "",
        "answer_sources": [],
        "verification_status": "not_verified",
        "verification_feedback": "",
        "error": None,
        "run_log": []
    }

    state["query"] = query
    result = graph.invoke(state)
    return result