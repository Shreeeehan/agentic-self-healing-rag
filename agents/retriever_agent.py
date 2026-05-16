"""
agents/retriever_agent.py

Job: Take the query from state, search ChromaDB, store results back in state.
This is the simplest agent — but everything depends on it.
"""

import os
from dotenv import load_dotenv
from core.vectorstore import get_vectorstore
from graph.state import RAGState, HealingStatus

load_dotenv()

TOP_K = int(os.getenv("TOP_K_RETRIEVAL", 5))

def retriever_agent(state: RAGState) -> RAGState:
    """
    LangGraph node — receives state, returns updated state.

    Uses rephrased_query if Repair Agent has rewritten it,
    otherwise uses the original query.
    """
    # Use rephrased query if available (after healing), else original
    query = state.rephrased_query if state.rephrased_query else state.query

    state.log(f"[Retriever] Searching for: '{query[:60]}'")

    try:
        vs = get_vectorstore()
        results = vs.search(query=query, top_k=TOP_K)

        state.retrieved_chunks = results
        state.healing_status = HealingStatus.NOT_STARTED

        state.log(f"[Retriever] Found {len(results)} chunks. "
                  f"Top score: {results[0]['score'] if results else 0}")

    except Exception as e:
        state.error = str(e)
        state.retrieved_chunks = []
        state.log(f"[Retriever] ERROR: {e}")

    return state