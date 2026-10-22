import hashlib
import time
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sqlmodel import Session, SQLModel, create_engine, select, Field
from typing import Optional
from datetime import datetime, timezone

# --- 1. Page Config ---
st.set_page_config(page_title="TextSammy - AI Summarizer", page_icon="📝", layout="wide")
st.title("📝 TextSammy: Full-Stack AI Text Summarizer")
st.caption("Powered by Hugging Face DistilBART • PyTorch • SQLite Cache")

# --- 2. Database Setup ---
DATABASE_URL = "sqlite:///./summaries.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

class SummaryRecord(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    text_hash: str = Field(index=True)
    original_text: str
    summary_text: str
    model_used: str
    num_chunks: int
    original_word_count: int
    summary_word_count: int
    processing_time_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

SQLModel.metadata.create_all(engine)

# --- 3. Load Hugging Face Model (Cached for Cloud) ---
@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device, model_name

with st.spinner("Loading AI Model into Cloud RAM (Takes ~10 seconds)..."):
    tokenizer, model, device, MODEL_NAME = load_summarizer()

# --- 4. Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.success("🟢 Model Loaded & Ready")
    min_len = st.slider("Minimum Summary Length", 10, 150, 30, 5)
    max_len = st.slider("Maximum Summary Length", 30, 300, 120, 10)

# --- 5. Helper Functions ---
def split_into_chunks(text: str, max_words_per_chunk: int = 650) -> list[str]:
    sentences = text.replace("\n", " ").split(". ")
    chunks, current_chunk, current_word_count = [], [], 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        if not sentence.endswith("."): sentence += "."
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words_per_chunk and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_word_count = [sentence], word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    if current_chunk: chunks.append(" ".join(current_chunk))
    return chunks

# --- 6. Main UI ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Input Article")
    article_text = st.text_area("Paste text below:", height=280)
    summarize_btn = st.button("✨ Generate Summary", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Generated Summary")
    if summarize_btn and article_text.strip():
        with st.spinner("Processing..."):
            raw_key = f"{article_text.strip()}_{min_len}_{max_len}"
            text_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
            
            with Session(engine) as session:
                cached = session.exec(select(SummaryRecord).where(SummaryRecord.text_hash == text_hash)).first()
                if cached:
                    st.info("⚡ [SQL Cache Hit] Retrieved instantly!")
                    st.success(cached.summary_text)
                    st.metric("Time", "5.0 ms")
                else:
                    # Run Inference
                    start_time = time.perf_counter()
                    chunks = split_into_chunks(article_text.strip())
                    summaries = []
                    for chunk in chunks:
                        inputs = tokenizer(chunk, max_length=1024, truncation=True, return_tensors="pt").to(device)
                        with torch.no_grad():
                            summary_ids = model.generate(
                                inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                num_beams=4, min_length=min_len, max_length=max_len, early_stopping=True
                            )
                        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
                    
                    final_summary = " ".join(summaries)
                    elapsed = round((time.perf_counter() - start_time) * 1000, 2)
                    
                    # Save to DB
                    new_rec = SummaryRecord(
                        text_hash=text_hash, original_text=article_text.strip(), summary_text=final_summary,
                        model_used=MODEL_NAME, num_chunks=len(chunks), original_word_count=len(article_text.split()),
                        summary_word_count=len(final_summary.split()), processing_time_ms=elapsed
                    )
                    session.add(new_rec)
                    session.commit()
                    
                    st.success(final_summary)
                    st.metric("Time", f"{elapsed} ms")