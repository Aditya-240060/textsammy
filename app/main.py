import hashlib
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from app.database import create_db_and_tables, get_session
from app.models import SummarizeRequest, SummaryRecord
from app.summarizer import TextSummarizer

# Dictionary to hold our loaded Deep Learning model in memory while the server stays alive
ml_context = {}


# 1. Lifespan Manager: Runs ONCE when FastAPI starts up, and cleans up when it shuts down
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[FastAPI Startup] Initializing SQL tables and loading Transformer model...")
    create_db_and_tables()
    ml_context["summarizer"] = TextSummarizer()
    print("[FastAPI Startup] Server is ready to accept requests!")
    yield
    # Anything after 'yield' runs when you press Ctrl+C to stop the server
    print("[FastAPI Shutdown] Clearing model from memory...")
    ml_context.clear()


app = FastAPI(
    title="TextSammy AI Summarizer API",
    description="Full-Stack NLP Summarizer using Hugging Face Transformers, FastAPI, and SQL",
    version="1.0.0",
    lifespan=lifespan,
)

# 2. CORS Middleware: Allows your Frontend UI (running on a different port) to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Health Check Endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "summarizer" in ml_context,
    }


# 4. Main Summarization Endpoint (POST /summarize)
@app.post("/summarize", response_model=SummaryRecord)
def create_summary(
    request: SummarizeRequest,
    session: Session = Depends(get_session),
):
    cleaned_text = request.text.strip()
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if request.min_length >= request.max_length:
        raise HTTPException(
            status_code=400,
            detail="min_length must be strictly smaller than max_length.",
        )

    # Create SHA-256 fingerprint of (text + min_length + max_length)
    raw_key = f"{cleaned_text}_{request.min_length}_{request.max_length}"
    text_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    # Step A: Check SQL Cache first
    statement = select(SummaryRecord).where(SummaryRecord.text_hash == text_hash)
    cached_record = session.exec(statement).first()

    if cached_record:
        print(f"⚡ [API CACHE HIT] Returning cached summary ID={cached_record.id}")
        return cached_record

    # Step B: Cache Miss -> Run Deep Learning Inference
    print("🧠 [API CACHE MISS] Running Transformer inference...")
    summarizer: TextSummarizer = ml_context["summarizer"]
    result = summarizer.summarize(
        text=cleaned_text,
        min_length=request.min_length,
        max_length=request.max_length,
    )

    # Step C: Save new record to SQL Database
    new_record = SummaryRecord(
        text_hash=text_hash,
        original_text=cleaned_text,
        **result,
    )
    session.add(new_record)
    session.commit()
    session.refresh(new_record)

    return new_record


# 5. History Endpoint (GET /history)
@app.get("/history", response_model=list[SummaryRecord])
def get_summary_history(
    limit: int = 10,
    session: Session = Depends(get_session),
):
    statement = (
        select(SummaryRecord)
        .order_by(SummaryRecord.created_at.desc())
        .limit(limit)
    )
    return session.exec(statement).all()


# 6. Delete Endpoint (DELETE /history/{record_id})
@app.delete("/history/{record_id}")
def delete_summary(
    record_id: int,
    session: Session = Depends(get_session),
):
    record = session.get(SummaryRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Summary record not found.")
    session.delete(record)
    session.commit()
    return {"deleted_id": record_id, "message": "Record deleted successfully."}