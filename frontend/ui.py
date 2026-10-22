import requests
import streamlit as st

# The URL where your FastAPI backend is running in Terminal #1
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="TextSammy - AI Summarizer",
    page_icon="📝",
    layout="wide",
)

st.title("📝 TextSammy: Full-Stack AI Text Summarizer")
st.caption("Powered by Hugging Face DistilBART • FastAPI • SQLModel (SQLite Cache)")

# --- 1. Sidebar: Backend Health Check & Model Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Check if FastAPI backend in Terminal #1 is alive
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=15)
        if health_resp.status_code == 200:
            st.success("🟢 FastAPI Backend: Online")
        else:
            st.warning("🟡 FastAPI Backend: Unhealthy")
    except requests.exceptions.ConnectionError:
        st.error("🔴 FastAPI Offline! Keep `uvicorn app.main:app --reload` running in Terminal 1.")

    st.divider()
    min_len = st.slider("Minimum Summary Length (tokens)", min_value=10, max_value=150, value=30, step=5)
    max_len = st.slider("Maximum Summary Length (tokens)", min_value=30, max_value=300, value=120, step=10)

# --- 2. Main Area: Input & Summarization ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Input Article")
    article_text = st.text_area(
        "Paste any long article, report, or lecture notes below:",
        height=280,
        placeholder="Paste your text here...",
    )
    summarize_btn = st.button("✨ Generate Summary", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Generated Summary")
    if summarize_btn:
        if not article_text.strip():
            st.warning("Please enter some text first.")
        elif min_len >= max_len:
            st.error("Minimum length must be strictly less than maximum length.")
        else:
            with st.spinner("Running Transformer / Checking SQL Cache..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/summarize",
                        json={
                            "text": article_text,
                            "min_length": min_len,
                            "max_length": max_len,
                        },
                        timeout=120,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.success(data["summary_text"])

                        # Calculate compression ratio
                        orig_words = data["original_word_count"]
                        sum_words = data["summary_word_count"]
                        reduction_pct = round((1 - (sum_words / max(1, orig_words))) * 100, 1)

                        # Display Engineering Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Original Words", orig_words)
                        m2.metric("Summary Words", sum_words, f"-{reduction_pct}%")
                        m3.metric("Chunks Used", data["num_chunks"])
                        m4.metric("Model Time", f"{data['processing_time_ms']} ms")
                    else:
                        error_detail = response.json().get("detail", "Unknown error")
                        st.error(f"API Error ({response.status_code}): {error_detail}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI. Make sure Terminal 1 is still running Uvicorn!")

# --- 3. SQL Database History Section ---
st.divider()
st.subheader("🗄️ Recent Summaries (Stored in SQLite Database)")

try:
    history_resp = requests.get(f"{API_BASE_URL}/history?limit=10", timeout=5)
    if history_resp.status_code == 200:
        records = history_resp.json()
        if not records:
            st.info("No summaries in the database yet. Generate one above!")
        else:
            for rec in records:
                with st.expander(
                    f"ID #{rec['id']} | {rec['original_word_count']} words ➔ {rec['summary_word_count']} words | {rec['created_at'][:19]}"
                ):
                    st.markdown(f"**Summary:** {rec['summary_text']}")
                    st.caption(
                        f"Model: `{rec['model_used']}` | Chunks: `{rec['num_chunks']}` | Inference Time: `{rec['processing_time_ms']} ms`"
                    )
                    if st.button(f"🗑️ Delete Record #{rec['id']}", key=f"del_{rec['id']}"):
                        requests.delete(f"{API_BASE_URL}/history/{rec['id']}")
                        st.rerun()
except requests.exceptions.ConnectionError:
    st.caption("Connect FastAPI backend to view SQL history.")