"""Data class representing an uploaded source document."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    doc_id: str
    filename: str
    num_chunks: int = 0
    status: str = "processing"  # processing | ready | failed
    error: Optional[str] = None
