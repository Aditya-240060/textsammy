from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, SQLModel


# 1. SQL Table Model (table=True means SQLModel creates a real SQL table for this)
class SummaryRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text_hash: str = Field(index=True)  # Indexed for fast O(log N) SQL cache lookups
    original_text: str
    summary_text: str
    model_used: str
    num_chunks: int
    original_word_count: int
    summary_word_count: int
    processing_time_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# 2. Pydantic Request Schema (table=False by default — used only to validate user input)
class SummarizeRequest(SQLModel):
    text: str
    min_length: int = Field(default=30, ge=10, le=200)
    max_length: int = Field(default=130, ge=20, le=500)