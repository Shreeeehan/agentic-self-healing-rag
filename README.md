# 🧠 Agentic Self-Healing RAG

A production-minded multi-agent RAG system that autonomously detects and repairs retrieval failures before they reach the user.

## 🎯 What Makes This Different

Most RAG systems retrieve → generate → done. This system adds a self-healing loop:

- **Critic Agent** scores retrieval quality using LLM-as-judge
- **Repair Agent** rewrites failed queries using 3 strategies (expansion, simplification, term replacement)
- **Verifier Agent** catches hallucinations before the answer reaches the user
- **Circuit breaker** prevents infinite loops (max 3 healing attempts)

## 🏗️ Architecture

```
User Query
    ↓
Retriever Agent  →  searches ChromaDB vector store
    ↓
Critic Agent     →  scores relevance (0.0 - 1.0)
    ↓
[score < 0.6]   →  Repair Agent rewrites query → loop back
[score ≥ 0.6]   →  Answer Generator
    ↓
Verifier Agent   →  hallucination check
    ↓
Final Answer + Sources
```

## 🤖 Agent Pipeline

| Agent | Job | Tech |
|---|---|---|
| Retriever | Semantic search over vector store | ChromaDB + HuggingFace embeddings |
| Critic | Score retrieval quality | LLM-as-judge (Groq Llama 3.3 70B) |
| Repair | Rewrite failed queries | LangChain prompt engineering |
| Generator | Generate cited answer | Groq Llama 3.3 70B |
| Verifier | Detect hallucinations | LLM-as-judge |

## 🛠️ Tech Stack

- **Orchestration** — LangGraph (stateful agent graphs)
- **LLM** — Groq (llama-3.3-70b-versatile)
- **Vector DB** — ChromaDB (local persistent store)
- **Embeddings** — HuggingFace all-MiniLM-L6-v2
- **PDF Parsing** — PyMuPDF (fitz)
- **API** — FastAPI + Uvicorn
- **UI** — Streamlit
- **Evaluation** — RAGAS

## 🚀 Getting Started

### 1. Clone and setup
```bash
git clone https://github.com/Shreeeehan/agentic-self-healing-rag
cd agentic-self-healing-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Add your Groq API key to .env
# Get free key at https://console.groq.com
```

### 3. Run Streamlit UI
```bash
python -m streamlit run ui/app.py
```

### 4. Run FastAPI
```bash
python -m uvicorn api.main:app --reload
# Visit http://127.0.0.1:8000/docs
```

## 📁 Project Structure

```
agentic-self-healing-rag/
├── core/
│   ├── ingestion.py      # PDF → chunks
│   └── vectorstore.py    # ChromaDB wrapper
├── agents/
│   ├── retriever_agent.py
│   ├── critic_agent.py
│   ├── repair_agent.py
│   └── verifier_agent.py
├── graph/
│   ├── state.py          # Shared agent state
│   └── rag_graph.py      # LangGraph pipeline
├── ui/
│   └── app.py            # Streamlit interface
├── api/
│   └── main.py           # FastAPI endpoints
└── evaluation/
    └── ragas_eval.py     # RAGAS metrics
```

## 🔑 Environment Variables

```bash
GROQ_API_KEY=           # Required — get at console.groq.com
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=5
CRITIC_SCORE_THRESHOLD=0.6
MAX_HEALING_LOOPS=3
```

## 🧪 Self-Healing in Action

When a query retrieves irrelevant chunks:
```
[Retriever] Found 5 chunks. Top score: -0.19
[Critic]    FAILED — score=0.00
[Router]    Critic failed → triggering repair (loop 1)
[Repair]    New query: 'attention mechanisms in machine translation'
[Retriever] Found 5 chunks. Top score: 0.54
[Critic]    FAILED — score=0.00
[Router]    Critic failed → triggering repair (loop 2)
[Repair]    New query: 'attention mechanisms for language translation'
...
[Router]    Max loops reached → generating best effort answer
[Verifier]  VERIFIED
```

## 📊 Responsible AI Features

- ✅ Source citations on every answer (filename + page number)
- ✅ Hallucination detection before answer delivery
- ✅ Explicit warnings when verification fails
- ✅ Full agent trace logged for every query
- ✅ Circuit breaker prevents runaway loops

## 👤 Author

**Shreehan Yedatkar**  
Built with LangGraph + ChromaDB + Groq