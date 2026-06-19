# RAG Chat

A production-leaning RAG application: Flask REST API + Streamlit chat UI, FAISS
for vector search, SQLite for chunk metadata, and GPT-4o / text-embedding-3-small
via the GitHub Models endpoint.

## Architecture

```
models/          Document, Chunk dataclasses (shared shapes)
backend/
  app.py           Flask API: /upload /query /documents /health
  config.py        env-driven settings
  ingestion.py     file parsing (pdf/docx/pptx/xlsx/csv/txt) + token chunking
  retrieval.py      query embedding + FAISS/SQLite retrieval
  generation.py    LLMService (GitHub Models: embeddings + chat)
  vector_store.py  FAISSWrapper (IndexIDMap + IndexFlatIP, pickle-persisted)
  metadata_store.py SQLiteMetadataStore (chunks + documents tables)
  cache.py         CacheService (Redis, falls back to in-memory dict)
  utils.py         logging, hashing, id helpers
frontend/
  streamlit_app.py  chat UI, SSE streaming client
```

The Flask API is stateless — all state is the FAISS pickle file and the SQLite
file under `backend/data/`. Any number of Flask worker processes can share
that directory.

## 1. Get a GitHub PAT for GitHub Models

1. github.com → Settings → Developer settings → Personal access tokens
2. Generate a token with the `models:read` permission (no other scopes needed)
3. Put it in `.env` as `GITHUB_PAT=...`

GitHub Models has rate limits on the free tier — if you see 429s under load,
slow down concurrent uploads or request a quota increase.

## 2. Run with Docker (recommended)

```bash
cp .env.example .env   # then fill in GITHUB_PAT
docker compose up --build
```

- Streamlit UI: http://localhost:8501
- Flask API: http://localhost:5000
- Redis: localhost:6379 (used automatically; the app also works if you remove this service — it'll fall back to an in-memory cache)

Data persists in the `rag_data` named volume (mounted at `/app/backend/data`).

## 3. Run locally without Docker

```bash
python -m venv venv && source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # fill in GITHUB_PAT
export $(grep -v '^#' .env | xargs)   # or use python-dotenv / set vars manually

# Terminal 1 — backend
cd backend && python app.py

# Terminal 2 — frontend
cd frontend && BACKEND_URL=http://localhost:5000 streamlit run streamlit_app.py
```

## 4. API reference

- `POST /upload` — multipart form, field `file`. Returns `{"doc_id", "num_chunks"}`.
- `POST /query` — JSON `{"question", "session_id", "filters": {"doc_id": "..."}}`.
  - Default: Server-Sent Events stream — lines `data: {"token": "..."}` followed
    by a final `data: {"done": true, "sources": [...]}`.
  - Pass `?stream=false` for a single JSON response `{"answer", "sources"}`
    (this path is also query-cached).
- `DELETE /documents/<doc_id>` — removes the doc's chunks from FAISS + SQLite.
- `GET /documents` — lists uploaded documents (used by the frontend's filter dropdown).
- `GET /health` — health check.

## 5. Scaling

- **Flask workers**: `gunicorn -w 4 -b 0.0.0.0:5000 app:app` (already the
  container's default command). Increase `-w` based on CPU cores; each worker
  loads its own copy of the FAISS index into memory, so watch RAM if the index
  grows large.
- **Behind Nginx**: put Nginx in front of the gunicorn workers as a reverse
  proxy / load balancer, e.g.:
  ```nginx
  upstream rag_backend {
      server backend1:5000;
      server backend2:5000;
  }
  server {
      listen 80;
      location / {
          proxy_pass http://rag_backend;
          proxy_set_header Host $host;
          proxy_buffering off;          # needed so SSE streams flow through
      }
  }
  ```
- **Multiple backend instances**: the current setup (pickle file + SQLite
  file) assumes a single writer. For true multi-instance scaling, point
  `FAISS_INDEX_PATH` / `SQLITE_DB_PATH` at a shared volume (NFS/EFS) for
  read-mostly workloads, or — better — swap `vector_store.py` for a managed
  vector DB (e.g. Qdrant, pgvector) and `metadata_store.py` for PostgreSQL.
  Both are isolated behind their current class interfaces, so call sites in
  `app.py` don't change.
- **Redis**: already used for the embedding/query cache when `REDIS_URL` is
  set; this also de-duplicates re-uploaded/identical chunks across instances.

## 6. Extending

- **Swap the LLM/embedding provider**: edit `generation.py` only — change the
  `base_url`/model names in `LLMService`, keep `get_embeddings` / `generate` /
  `build_messages` signatures the same.
- **Swap the vector store**: implement `add(vectors, ids)`, `search(query_vector, k)`,
  `delete(ids)` against your new backend, matching `FAISSWrapper`'s interface.
- **Swap metadata storage**: implement the same method names as
  `SQLiteMetadataStore` against Postgres/Redis.
- **Add auth**: add an `@app.before_request` check or a proper auth library;
  the API has none currently.
- **Persistent conversation memory**: currently per-browser-session in
  Streamlit only. To persist across sessions, add a `conversations` table
  keyed by `session_id` in `metadata_store.py` and pass history into
  `build_messages`.

## Notes / known simplifications

- Embeddings are cached by SHA256 of chunk text (skips re-embedding identical
  content); query caching only applies to the non-streaming (`?stream=false`)
  path, since caching partial token streams isn't meaningful.
- Filtering by `doc_id` over-fetches from FAISS (`k * 3`) then filters and
  truncates to `k` — simple and fine for moderate corpus sizes; for very large,
  heavily-filtered corpora, consider per-document FAISS indexes instead.
