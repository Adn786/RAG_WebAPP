"""Query embedding (with caching) and FAISS + SQLite retrieval."""
from utils import sha256_hash, setup_logger

logger = setup_logger(__name__)


def embed_query(llm_service, question: str, cache):
    """Embeds the user question, reusing a cached vector when available."""
    key = sha256_hash(question)
    cached = cache.get_embedding(key)
    if cached is not None:
        return cached
    vector = llm_service.get_embeddings([question])[0]
    cache.set_embedding(key, vector)
    return vector


def retrieve_context(vector_store, metadata_store, query_vector, k=10, doc_id_filter=None):
    """Searches FAISS, fetches chunk rows from SQLite, optionally filters by doc_id,
    and returns the top-k chunk dicts ordered by descending similarity."""
    # Over-fetch when filtering by document so we still end up with k results
    # after the filter is applied.
    search_k = k * 3 if doc_id_filter else k
    hits = vector_store.search(query_vector, k=search_k)
    if not hits:
        return []

    int_ids = [h[0] for h in hits]
    scores_by_id = {h[0]: h[1] for h in hits}

    rows = metadata_store.get_chunks_by_int_ids(int_ids)
    if doc_id_filter:
        rows = [r for r in rows if r["doc_id"] == doc_id_filter]

    rows.sort(key=lambda r: scores_by_id.get(r["int_id"], 0.0), reverse=True)
    rows = rows[:k]

    results = []
    for i, r in enumerate(rows, start=1):
        results.append({
            "ref": i,
            "text": r["text"],
            "source": r["source"],
            "page": r.get("page"),
            "doc_id": r["doc_id"],
            "score": scores_by_id.get(r["int_id"], 0.0),
        })
    return results
