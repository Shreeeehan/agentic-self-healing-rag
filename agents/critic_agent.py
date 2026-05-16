import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from graph.state import RAGState, HealingStatus

load_dotenv()

THRESHOLD = float(os.getenv("CRITIC_SCORE_THRESHOLD", 0.6))
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))

CRITIC_PROMPT = ChatPromptTemplate.from_template("""
You are a retrieval quality judge for a RAG system.

User Query: {query}

Retrieved Chunks:
{chunks}

Your job:
1. Score how relevant these chunks are to the query (0.0 to 1.0)
2. Give a one-line reason

Rules:
- 0.8-1.0 = highly relevant, directly answers the query
- 0.6-0.8 = mostly relevant, partial answer
- 0.4-0.6 = loosely related, missing key info
- 0.0-0.4 = irrelevant, wrong topic entirely

Respond in EXACTLY this format (no extra text):
SCORE: <number>
REASON: <one line explanation>
""")

def _format_chunks(chunks: list) -> str:
    if not chunks:
        return "No chunks retrieved."
    formatted = []
    for i, chunk in enumerate(chunks[:3]):
        formatted.append(
            f"Chunk {i+1} (score={chunk['score']}, "
            f"source={chunk['source']}, page={chunk['page']}):\n"
            f"{chunk['text'][:300]}..."
        )
    return "\n\n".join(formatted)

def _parse_critic_response(response: str) -> tuple:
    score = 0.0
    reason = "Could not parse response"
    for line in response.strip().split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                score = 0.0
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()
    return score, reason

def critic_agent(state: RAGState) -> RAGState:
    state.log("[Critic] Evaluating retrieval quality...")

    if not state.retrieved_chunks:
        state.critic_score = 0.0
        state.critic_feedback = "No chunks retrieved"
        state.critic_passed = False
        state.healing_status = HealingStatus.RETRIEVAL_BAD
        state.log("[Critic] FAILED — no chunks found")
        return state

    try:
        llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        chain = CRITIC_PROMPT | llm
        response = chain.invoke({
            "query": state.query,
            "chunks": _format_chunks(state.retrieved_chunks)
        })
        score, reason = _parse_critic_response(response.content)
        state.critic_score = score
        state.critic_feedback = reason

        if score >= THRESHOLD:
            state.critic_passed = True
            state.healing_status = HealingStatus.RETRIEVAL_OK
            state.log(f"[Critic] PASSED — score={score:.2f} | {reason}")
        else:
            state.critic_passed = False
            state.healing_status = HealingStatus.RETRIEVAL_BAD
            state.log(f"[Critic] FAILED — score={score:.2f} | {reason}")

    except Exception as e:
        state.error = str(e)
        state.critic_passed = False
        state.log(f"[Critic] ERROR: {e}")

    return state