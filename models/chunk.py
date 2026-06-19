"""Data class representing a single text chunk produced from a document."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    id: str          # UUID string, primary key in SQLite
    int_id: int      # deterministic int64 derived from id, used as the FAISS vector id
    doc_id: str
    text: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None
    embedding_hash: Optional[str] = None
