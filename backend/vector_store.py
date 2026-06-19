"""Thin wrapper around a FAISS IndexIDMap(IndexFlatIP) for cosine-similarity search.

Vectors are L2-normalized before insertion/search so that inner product is
equivalent to cosine similarity. The index is persisted to disk via pickle
after every mutation so it survives process restarts (single-instance dev
setup; see README for multi-instance notes).
"""
import os
import pickle
import threading

import faiss
import numpy as np

from utils import setup_logger

logger = setup_logger(__name__)


class FAISSWrapper:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self._lock = threading.Lock()
        self.index = self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    index = pickle.load(f)
                logger.info("Loaded FAISS index from %s (%d vectors)", self.index_path, index.ntotal)
                return index
            except Exception as exc:
                logger.warning("Failed to load FAISS index (%s); creating a new one", exc)
        base = faiss.IndexFlatIP(self.dim)
        return faiss.IndexIDMap(base)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def add(self, vectors, ids):
        """vectors: list[list[float]] ; ids: list[int] (int64)."""
        with self._lock:
            arr = self._normalize(np.array(vectors, dtype="float32"))
            id_arr = np.array(ids, dtype="int64")
            self.index.add_with_ids(arr, id_arr)
            self._save()

    def search(self, query_vector, k: int = 10):
        """Returns a list of (int_id, score) sorted by descending similarity."""
        with self._lock:
            if self.index.ntotal == 0:
                return []
            q = self._normalize(np.array([query_vector], dtype="float32"))
            scores, ids = self.index.search(q, min(k, self.index.ntotal))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def delete(self, ids):
        with self._lock:
            try:
                self.index.remove_ids(np.array(ids, dtype="int64"))
                self._save()
            except Exception as exc:
                logger.warning("FAISS delete failed: %s", exc)

    def _save(self):
        try:
            tmp_path = self.index_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(self.index, f)
            os.replace(tmp_path, self.index_path)
        except Exception as exc:
            logger.error("Failed to persist FAISS index: %s", exc)
