"""SQLite-backed repository for chunk and document metadata.

Kept behind this class so the storage backend can later be swapped for
Redis/PostgreSQL without touching call sites in app.py.
"""
import sqlite3
import threading
import time
from contextlib import contextmanager

from utils import setup_logger

logger = setup_logger(__name__)


class SQLiteMetadataStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    int_id INTEGER NOT NULL,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page INTEGER,
                    source TEXT,
                    section TEXT,
                    embedding_hash TEXT,
                    created_at REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_int_id ON chunks(int_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    num_chunks INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'processing',
                    error TEXT,
                    created_at REAL
                )
                """
            )

    # --- chunks -----------------------------------------------------------
    def insert_chunks(self, chunks):
        """chunks: list of dicts with keys id, int_id, doc_id, text, page, source,
        section, embedding_hash."""
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (id, int_id, doc_id, text, page, source, section, embedding_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c["id"], c["int_id"], c["doc_id"], c["text"], c.get("page"),
                        c.get("source"), c.get("section"), c.get("embedding_hash"), time.time(),
                    )
                    for c in chunks
                ],
            )

    def get_chunks_by_int_ids(self, int_ids):
        if not int_ids:
            return []
        with self._connect() as conn:
            placeholders = ",".join("?" * len(int_ids))
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE int_id IN ({placeholders})", int_ids
            ).fetchall()
            return [dict(r) for r in rows]

    def get_chunks_by_doc_id(self, doc_id):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
            return [dict(r) for r in rows]

    def delete_by_doc_id(self, doc_id):
        """Deletes chunk rows for a document and returns their int_ids so the
        caller can also remove them from the FAISS index."""
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT int_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
            int_ids = [r["int_id"] for r in rows]
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            return int_ids

    # --- documents ----------------------------------------------------------
    def upsert_document(self, doc_id, filename, num_chunks=0, status="processing", error=None):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (doc_id, filename, num_chunks, status, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    num_chunks = excluded.num_chunks,
                    status = excluded.status,
                    error = excluded.error
                """,
                (doc_id, filename, num_chunks, status, error, time.time()),
            )

    def get_document(self, doc_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def list_documents(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, doc_id):
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
