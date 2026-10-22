import hashlib
from sqlmodel import Session, select

from app.database import create_db_and_tables, engine
from app.models import SummaryRecord
from app.summarizer import TextSummarizer

# 1. Create the SQLite database file (summaries.db) and table
create_db_and_tables()
summarizer_service = TextSummarizer()

article = """
Artificial Intelligence engineering goes far beyond training a model in a notebook. 
In real production systems, input data is unpredictable. Users might submit a single 
short sentence or a massive ten-page report. If an engineer does not handle token 
limits by splitting text along sentence boundaries, the Transformer model will silently 
ignore everything past the first thousand tokens. Furthermore, tracking latency and 
word count metrics allows teams to monitor model performance in a SQL database.
"""

# Create a unique fingerprint of (text + min_length + max_length)
min_len, max_len = 20, 60
raw_key = f"{article.strip()}_{min_len}_{max_len}"
text_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

with Session(engine) as session:
    # 2. Check SQL Cache first!
    statement = select(SummaryRecord).where(SummaryRecord.text_hash == text_hash)
    existing_record = session.exec(statement).first()

    if existing_record:
        print("\n⚡ [SQL CACHE HIT] Found in database instantly! Skipping Deep Learning model.")
        print(f"Database ID: {existing_record.id}")
        print(f"Summary: {existing_record.summary_text}")
    else:
        print("\n🧠 [SQL CACHE MISS] Running Hugging Face Transformer...")
        result = summarizer_service.summarize(article, min_length=min_len, max_length=max_len)

        # 3. Save result into SQL Database
        new_record = SummaryRecord(
            text_hash=text_hash,
            original_text=article.strip(),
            **result
        )
        session.add(new_record)
        session.commit()
        session.refresh(new_record)

        print(f"Saved to SQL Database with ID: {new_record.id} (Took {new_record.processing_time_ms} ms)")
        print(f"Summary: {new_record.summary_text}")