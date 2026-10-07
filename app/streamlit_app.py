import sys
from pathlib import Path

# Ensure project root is in sys.path when running from any working directory
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import json
import time
import pandas as pd

from src.rag_pipeline import FinSecureRAG
from evals.compare import compare_eval_runs, format_markdown_diff_table

st.set_page_config(
    page_title="FinSecure RAG | Production SEC Assistant & Evaluation Platform",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
BASELINE_FILE = BASE_DIR / "baselines" / "baseline.json"
ONLINE_LOGS_FILE = BASE_DIR / "data" / "online_eval_results.json"
FLYWHEEL_FILE = BASE_DIR / "goldens" / "curated_flywheel_cases.json"

st.title("📈 FinSecure RAG: Enterprise SEC Financial Assistant & Evaluation Suite")
st.caption("End-to-End RAG Architecture + 3-Tier Multi-Level Evaluation + CI/CD Regression Engine")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation Mode",
    [
        "💬 Interactive SEC Analyst Assistant",
        "📊 3-Tier Evaluation Dashboard",
        "🛡️ CI/CD Regression Gatekeeper",
        "🔄 Online Observability & Data Flywheel"
    ]
)

# Initialize RAG Pipeline in session state
@st.cache_resource
def get_rag_pipeline():
    return FinSecureRAG(top_k=3)

rag = get_rag_pipeline()

# ----------------------------------------------------
# PAGE 1: INTERACTIVE SEC ANALYST ASSISTANT
# ----------------------------------------------------
if page == "💬 Interactive SEC Analyst Assistant":
    st.subheader("Interactive Financial Query Interface")
    st.markdown("""
    Ask questions regarding **NVIDIA (NVDA Q3 FY25)**, **Apple (AAPL Q4 FY24)**, or **Microsoft (MSFT Q1 FY25)** earnings calls.
    Answers are constrained by prompt-hardened SEC safeguards.
    """)

    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input(
            "Enter Financial Analyst Query:",
            value="What was NVIDIA's total revenue in Q3 FY25 and what did Jensen say about Blackwell demand?"
        )
    with col2:
        top_k_select = st.slider("Retrieval Top-K", min_value=1, max_value=6, value=3)

    if st.button("🚀 Execute RAG Query", type="primary"):
        with st.spinner("Retrieving filings and generating answer..."):
            rag.top_k = top_k_select
            res = rag.query(user_query)

        st.success(f"Response generated in {res['metadata']['total_latency_sec']:.2f}s (Retrieval: {res['metadata']['retrieval_latency_sec']:.2f}s | Generation: {res['metadata']['generation_latency_sec']:.2f}s)")

        st.markdown("### 📝 Grounded Response:")
        st.write(res["answer"])

        st.markdown("### 🔍 Retrieved SEC Filing Contexts:")
        for idx, doc in enumerate(res["retrieved_docs"], 1):
            with st.expander(f"Chunk #{idx} | Company: {doc.get('metadata', {}).get('company')} | Hybrid Score: {doc.get('score', 0):.4f}"):
                st.write(doc["page_content"])

# ----------------------------------------------------
# PAGE 2: 3-TIER EVALUATION DASHBOARD
# ----------------------------------------------------
elif page == "📊 3-Tier Evaluation Dashboard":
    st.subheader("Comprehensive 3-Tier Evaluation Telemetry")
    if not BASELINE_FILE.exists():
        st.error("Baseline report not found. Run `python -m evals.run_suite --output baselines/baseline.json` first.")
    else:
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            base_data = json.load(f)

        metrics = base_data.get("metrics", {})
        
        st.markdown("#### 1. Level 1: Component in Isolation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retriever Recall", f"{metrics.get('retriever_recall', 0):.1%}")
        m2.metric("Retriever MRR", f"{metrics.get('retriever_mrr', 0):.2f}")
        m3.metric("Generator Faithfulness", f"{metrics.get('generator_faithfulness', 0):.1%}")
        m4.metric("Generator Relevancy", f"{metrics.get('generator_relevancy', 0):.1%}")

        st.markdown("#### 2. Level 2: Pipeline RAG Triad")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Contextual Relevancy", f"{metrics.get('pipeline_contextual_relevancy', 0):.1%}")
        m6.metric("Pipeline Faithfulness", f"{metrics.get('pipeline_faithfulness', 0):.1%}")
        m7.metric("Answer Relevancy", f"{metrics.get('pipeline_answer_relevancy', 0):.1%}")
        m8.metric("Intra-Chunk Noise", f"{metrics.get('pipeline_intra_chunk_noise', 0):.1%}")

        st.markdown("#### 3. Level 3: Application Quality & Safety")
        m9, m10, m11, m12 = st.columns(4)
        m9.metric("Financial Correctness", f"{metrics.get('quality_correctness', 0):.1%}")
        m10.metric("Completeness", f"{metrics.get('quality_completeness', 0):.1%}")
        m11.metric("Guardrail Refusal Rate", f"{metrics.get('safety_refusal_rate', 0):.1%}")
        m12.metric("MNPI Leakage Rate", f"{metrics.get('safety_mnpi_leakage', 0):.1%}")

        st.markdown("#### 4. Level 3: Operations & Token Economics")
        m13, m14, m15, m16 = st.columns(4)
        m13.metric("P95 Latency", f"{metrics.get('ops_p95_latency', 0):.2f}s")
        m14.metric("Mean TTFT", f"{metrics.get('ops_mean_ttft', 0):.2f}s")
        m15.metric("Cost / 1k Queries", f"${metrics.get('ops_cost_per_1k', 0):.4f}")
        m16.metric("Throughput (RPS)", f"{metrics.get('ops_rps', 0):.2f} req/s")

# ----------------------------------------------------
# PAGE 3: CI/CD REGRESSION GATEKEEPER
# ----------------------------------------------------
elif page == "🛡️ CI/CD Regression Gatekeeper":
    st.subheader("CI/CD Regression Prevention Engine")
    st.markdown("Compare Candidate System Run against the Established Production Baseline with strict ±2σ statistical noise margin enforcement.")

    c1, c2 = st.columns(2)
    with c1:
        base_path_input = st.text_input("Baseline File", value=str(BASELINE_FILE))
    with c2:
        cand_path_input = st.text_input("Candidate File", value=str(BASELINE_FILE))

    if st.button("⚖️ Run Regression Analysis", type="primary"):
        b_p = Path(base_path_input)
        c_p = Path(cand_path_input)
        if not b_p.exists() or not c_p.exists():
            st.error("Please ensure both baseline and candidate files exist.")
        else:
            passed, report = compare_eval_runs(b_p, c_p)
            if passed:
                st.success("✅ **CI/CD PASS**: Candidate satisfies all critical quality, safety, and operational thresholds.")
            else:
                st.error("🚨 **CI/CD BLOCKED**: Critical regression detected! Pull request cannot be merged.")

            df = pd.DataFrame(report["comparisons"])
            st.dataframe(df, use_container_width=True)

# ----------------------------------------------------
# PAGE 4: ONLINE OBSERVABILITY & FLYWHEEL
# ----------------------------------------------------
elif page == "🔄 Online Observability & Data Flywheel":
    st.subheader("Session 8: Online Observability & Autonomous Flywheel")
    st.markdown("Real-time telemetry and continuous dataset synthesis from production edge cases.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📡 Live Production Audit Logs")
        if ONLINE_LOGS_FILE.exists():
            with open(ONLINE_LOGS_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
            st.json(logs)
        else:
            st.info("No online trace logs found. Run `python -m evals.eval_online` to generate logs.")

    with col2:
        st.markdown("### 🔄 Synthesized Flywheel Benchmark Cases")
        if FLYWHEEL_FILE.exists():
            with open(FLYWHEEL_FILE, "r", encoding="utf-8") as f:
                flywheel = json.load(f)
            st.json(flywheel)
        else:
            st.info("No flywheel cases generated yet. Run `python -m evals.flywheel`.")
