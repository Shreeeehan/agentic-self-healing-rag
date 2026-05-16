"""
api/main.py

FastAPI REST API — exposes the RAG pipeline as HTTP endpoints.
This is what a React frontend or any external app would call.

Run with: uvicorn api.main:app --reload
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil

from graph.rag_graph import run_pipeline
from core.ingestion import ingest_pdf
from core.vectorstore import get_vectorstore

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic Self-Healing RAG",
    description="Multi-agent RAG pipeline with autonomous retrieval repair",
    version="1.0.0"
)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response Models ───────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    critic_score: float
    critic_passed: bool
    critic_feedback: str
    healing_loop_count: int
    verification_status: str
    verification_feedback: str
    rephrased_query: Optional[str]
    sources: List[dict]
    run_log: List[str]
    error: Optional[str]

class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str

class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    persist_dir: str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Agentic Self-Healing RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/query", "/ingest", "/stats", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Run the full multi-agent pipeline on a query.
    Returns the answer + full agent trace.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = run_pipeline(request.query)
        return QueryResponse(
            query=request.query,
            final_answer=result.get("final_answer", ""),
            critic_score=result.get("critic_score", 0.0),
            critic_passed=result.get("critic_passed", False),
            critic_feedback=result.get("critic_feedback", ""),
            healing_loop_count=result.get("healing_loop_count", 0),
            verification_status=result.get("verification_status", "not_verified"),
            verification_feedback=result.get("verification_feedback", ""),
            rephrased_query=result.get("rephrased_query", ""),
            sources=result.get("answer_sources", []),
            run_log=result.get("run_log", []),
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF into the vector store.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    temp_path = f"/tmp/{file.filename}"
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Ingest
        chunks = ingest_pdf(temp_path)
        vs = get_vectorstore()
        vs.add_chunks(chunks)

        return IngestResponse(
            filename=file.filename,
            chunks_created=len(chunks),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/stats", response_model=StatsResponse)
def stats_endpoint():
    """
    Get vector store statistics.
    """
    try:
        vs = get_vectorstore()
        stats = vs.get_collection_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/reset")
def reset_endpoint():
    """
    Wipe the vector store. Use when re-ingesting fresh documents.
    """
    try:
        vs = get_vectorstore()
        vs.delete_collection()
        return {"status": "success", "message": "Vector store reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))