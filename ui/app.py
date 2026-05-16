"""
ui/app.py

Streamlit UI for the Agentic Self-Healing RAG system.
This is what the user sees and interacts with.

Run with: streamlit run ui/app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from graph.rag_graph import run_pipeline, build_rag_graph
from core.ingestion import ingest_pdf, ingest_directory
from core.vectorstore import get_vectorstore

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Self-Healing RAG",
    page_icon="🧠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a0f; }
    .stApp { background-color: #0a0a0f; color: #e8e8f0; }
    .metric-card {
        background: #111118;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .agent-log {
        background: #0d1117;
        border: 1px solid #2a2a3a;
        border-radius: 6px;
        padding: 12px;
        font-family: monospace;
        font-size: 12px;
        max-height: 300px;
        overflow-y: auto;
    }
    .status-pass { color: #00ff88; font-weight: bold; }
    .status-fail { color: #ff6b35; font-weight: bold; }
    .status-heal { color: #7b61ff; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Agentic Self-Healing RAG")
st.markdown("*Multi-agent pipeline with autonomous retrieval repair*")
st.divider()

# ── Sidebar — Document Upload ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Knowledge Base")
    st.markdown("Upload PDFs to add to the vector store.")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", type="primary"):
            vs = get_vectorstore()
            total_chunks = 0

            with st.spinner("Ingesting documents..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Ingest
                    chunks = ingest_pdf(temp_path)
                    vs.add_chunks(chunks)
                    total_chunks += len(chunks)
                    os.remove(temp_path)

            st.success(f"✅ Ingested {len(uploaded_files)} file(s) → {total_chunks} chunks")

    st.divider()

    # Vector store stats
    st.markdown("## 📊 Vector Store Stats")
    try:
        vs = get_vectorstore()
        stats = vs.get_collection_stats()
        st.metric("Total Chunks", stats["total_chunks"])
        st.metric("Collection", stats["collection_name"])
    except Exception as e:
        st.error(f"VectorStore error: {e}")

    st.divider()
    st.markdown("## ⚙️ Settings")
    show_agent_trace = st.toggle("Show Agent Trace", value=True)
    show_sources = st.toggle("Show Sources", value=True)

# ── Main — Query Interface ────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 💬 Ask a Question")
    query = st.text_area(
        "Enter your query",
        placeholder="e.g. What is gradient descent? How does attention mechanism work?",
        height=100
    )

    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

with col2:
    st.markdown("## 🔄 Pipeline Status")
    status_placeholder = st.empty()
    status_placeholder.info("Waiting for query...")

# ── Run Pipeline ──────────────────────────────────────────────────────────────
if run_btn and query.strip():
    # Clear previous results
    status_placeholder.warning("🔄 Pipeline running...")

    with st.spinner("Agents working..."):
        try:
            result = run_pipeline(query)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 📋 Results")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        score = result.get("critic_score", 0)
        st.metric(
            "Critic Score",
            f"{score:.2f}",
            delta="PASSED" if result.get("critic_passed") else "FAILED"
        )

    with m2:
        loops = result.get("healing_loop_count", 0)
        st.metric(
            "Healing Loops",
            loops,
            delta="Self-healed" if loops > 0 else "No healing needed"
        )

    with m3:
        v_status = result.get("verification_status", "not_verified")
        st.metric(
            "Verification",
            v_status.upper(),
        )

    with m4:
        chunks = len(result.get("retrieved_chunks", []))
        st.metric("Chunks Retrieved", chunks)

    # Update status
    healing_loops = result.get("healing_loop_count", 0)
    if result.get("critic_passed"):
        status_placeholder.success("✅ Pipeline complete")
    elif healing_loops > 0:
        status_placeholder.warning(f"🔧 Healed {healing_loops}x then completed")
    else:
        status_placeholder.error("⚠️ Could not verify retrieval")

    # ── Final Answer ──────────────────────────────────────────────────────────
    st.markdown("### 💡 Answer")
    v_status = result.get("verification_status", "")

    if v_status == "hallucinated":
        st.warning("⚠️ Verifier flagged potential hallucination in this answer.")

    st.markdown(result.get("final_answer", "No answer generated."))

    # ── Rephrased Query (if healing happened) ─────────────────────────────────
    if result.get("rephrased_query"):
        with st.expander("🔧 Query Repair Details"):
            st.markdown(f"**Original:** {query}")
            st.markdown(f"**Rephrased:** {result.get('rephrased_query')}")
            st.markdown(f"**Healing loops:** {result.get('healing_loop_count')}")
            st.markdown(f"**Critic feedback:** {result.get('critic_feedback')}")

    # ── Sources ───────────────────────────────────────────────────────────────
    if show_sources and result.get("answer_sources"):
        with st.expander("📚 Sources"):
            for i, source in enumerate(result["answer_sources"]):
                st.markdown(
                    f"**{i+1}.** `{source['source']}` — "
                    f"Page {source['page']} | "
                    f"Score: {source['score']:.3f}"
                )

    # ── Agent Trace ───────────────────────────────────────────────────────────
    if show_agent_trace and result.get("run_log"):
        with st.expander("🔍 Agent Trace (Full Pipeline Log)", expanded=True):
            log_text = "\n".join(result["run_log"])
            st.code(log_text, language="bash")

elif run_btn and not query.strip():
    st.warning("Please enter a query first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Agentic Self-Healing RAG • "
    "LangGraph + ChromaDB + Groq • Built from scratch</small></center>",
    unsafe_allow_html=True
)