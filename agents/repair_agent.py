"""
agents/repair_agent.py

Job: When Critic fails, rewrite the query to improve retrieval.
     This is the actual "healing" step.

Strategies used:
- Query expansion (add related terms)
- Query decomposition (break complex query into simpler one)
- Keyword extraction (strip fluff, keep core terms)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from graph.state import RAGState, HealingStatus

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
MAX_LOOPS = int(os.getenv("MAX_HEALING_LOOPS", 3))

REPAIR_PROMPT = ChatPromptTemplate.from_template("""
You are a query repair specialist for a RAG system.

Original Query: {query}
Previous Rephrased Query: {rephrased_query}
Critic Feedback: {critic_feedback}
Critic Score: {critic_score}
Healing Attempt: {loop_count} of {max_loops}

The retrieval failed because the query didn't match well with the documents.
Rewrite the query to improve retrieval using one of these strategies:
- If query is vague: make it more specific
- If query is complex: simplify to core concept
- If query uses jargon: use simpler alternative terms
- If query is too narrow: expand with related terms

Respond with ONLY the rewritten query. No explanation. No quotes.
""")

def repair_agent(state: RAGState) -> RAGState:
    """
    LangGraph node — rewrites the query to improve retrieval.
    Increments healing_loop_count each time.
    """
    state.log(f"[Repair] Healing attempt {state.healing_loop_count + 1}/{MAX_LOOPS}")
    state.log(f"[Repair] Critic feedback: {state.critic_feedback}")

    # Check if we've hit the max healing loops
    if state.healing_loop_count >= MAX_LOOPS:
        state.healing_status = HealingStatus.FAILED
        state.log("[Repair] Max healing loops reached. Giving up.")
        return state

    try:
        llm = ChatGroq(model=LLM_MODEL, temperature=0.3)  # slight creativity for repair
        chain = REPAIR_PROMPT | llm

        response = chain.invoke({
            "query": state.query,
            "rephrased_query": state.rephrased_query or "None",
            "critic_feedback": state.critic_feedback,
            "critic_score": state.critic_score,
            "loop_count": state.healing_loop_count + 1,
            "max_loops": MAX_LOOPS
        })

        new_query = response.content.strip()
        state.rephrased_query = new_query
        state.healing_loop_count += 1
        state.healing_status = HealingStatus.HEALING

        state.log(f"[Repair] New query: '{new_query[:80]}'")

    except Exception as e:
        state.error = str(e)
        state.healing_status = HealingStatus.FAILED
        state.log(f"[Repair] ERROR: {e}")

    return state