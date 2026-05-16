"""
agents/verifier_agent.py

Job: Check if the final answer is grounded in the retrieved chunks.
     Catches hallucinations before the answer reaches the user.

This is the last quality gate before output.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from graph.state import RAGState, VerificationStatus

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))

VERIFIER_PROMPT = ChatPromptTemplate.from_template("""
You are a hallucination detector for a RAG system.

User Query: {query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Check if the answer is fully grounded in the retrieved context.

Rules:
- VERIFIED: Every claim in the answer is supported by the context
- HALLUCINATED: Answer contains claims NOT found in the context

Respond in EXACTLY this format:
STATUS: <VERIFIED or HALLUCINATED>
REASON: <one line explanation>
""")

def _format_context(chunks: list) -> str:
    if not chunks:
        return "No context available."
    return "\n\n".join([
        f"[Source: {c['source']}, Page: {c['page']}]\n{c['text'][:400]}"
        for c in chunks[:3]
    ])

def verifier_agent(state: RAGState) -> RAGState:
    """
    LangGraph node — verifies the answer against retrieved chunks.
    """
    state.log("[Verifier] Checking answer for hallucinations...")

    if not state.final_answer:
        state.verification_status = VerificationStatus.NOT_VERIFIED
        state.log("[Verifier] No answer to verify.")
        return state

    try:
        llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        chain = VERIFIER_PROMPT | llm

        response = chain.invoke({
            "query": state.query,
            "context": _format_context(state.retrieved_chunks),
            "answer": state.final_answer
        })

        result = response.content.strip()
        status = VerificationStatus.NOT_VERIFIED
        reason = ""

        for line in result.split("\n"):
            if line.startswith("STATUS:"):
                val = line.replace("STATUS:", "").strip().upper()
                if val == "VERIFIED":
                    status = VerificationStatus.VERIFIED
                elif val == "HALLUCINATED":
                    status = VerificationStatus.HALLUCINATED
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        state.verification_status = status
        state.verification_feedback = reason
        state.log(f"[Verifier] {status.value.upper()} — {reason}")

    except Exception as e:
        state.error = str(e)
        state.log(f"[Verifier] ERROR: {e}")

    return state