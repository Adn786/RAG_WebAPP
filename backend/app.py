"""Flask REST API for the RAG application. Stateless: all data lives in the
FAISS index file and SQLite DB under Config.DATA_DIR, so any number of worker
processes (gunicorn -w N) can serve requests safely as long as they share
that data directory (or, for multi-instance deployments, a shared volume)."""
import hashlib
import json
import os
import tempfile

from flask import Flask, Response, jsonify, request, stream_with_context

from cache import CacheService
from config import Config
from generation import LLMService
from ingestion import chunk_sections, parse_file
from metadata_store import SQLiteMetadataStore
from retrieval import embed_query, retrieve_context
from utils import allowed_file, new_id, setup_logger
from vector_store import FAISSWrapper

logger = setup_logger(__name__)

Config.validate()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_FILE_SIZE_MB * 1024 * 1024

vector_store = FAISSWrapper(dim=Config.EMBEDDING_DIM, index_path=Config.FAISS_INDEX_PATH)
metadata_store = SQLiteMetadataStore(db_path=Config.SQLITE_DB_PATH)
cache = CacheService(redis_url=Config.REDIS_URL)
llm_service = LLMService()


def _query_cache_key(question: str, doc_id_filter: str) -> str:
    normalized = question.strip().lower()
    raw = f"{normalized}|{doc_id_filter or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/documents", methods=["GET"])
def list_documents():
    return jsonify({"documents": metadata_store.list_documents()}), 200


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename, Config.ALLOWED_EXTENSIONS):
        return jsonify({"error": f"Unsupported file type: {file.filename}"}), 400

    doc_id = new_id()
    metadata_store.upsert_document(doc_id, file.filename, status="processing")

    tmp_path = None
    try:
        suffix = "." + file.filename.rsplit(".", 1)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        sections = parse_file(tmp_path, file.filename)
        chunks = chunk_sections(
            sections, doc_id, source=file.filename,
            chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP,
        )
        if not chunks:
            raise ValueError("Document produced no usable chunks")

        # Embed only chunks not already present in the embedding cache.
        vectors = [None] * len(chunks)
        to_embed_texts, to_embed_idx = [], []
        for i, c in enumerate(chunks):
            cached_vec = cache.get_embedding(c["embedding_hash"])
            if cached_vec is not None:
                vectors[i] = cached_vec
            else:
                to_embed_texts.append(c["text"])
                to_embed_idx.append(i)

        if to_embed_texts:
            new_vectors = llm_service.get_embeddings(to_embed_texts)
            for idx, vec in zip(to_embed_idx, new_vectors):
                vectors[idx] = vec
                cache.set_embedding(chunks[idx]["embedding_hash"], vec)

        vector_store.add(vectors, [c["int_id"] for c in chunks])
        metadata_store.insert_chunks(chunks)
        metadata_store.upsert_document(doc_id, file.filename, num_chunks=len(chunks), status="ready")

        return jsonify({"doc_id": doc_id, "num_chunks": len(chunks)}), 200

    except ValueError as exc:
        logger.warning("Upload validation error for %s: %s", file.filename, exc)
        metadata_store.upsert_document(doc_id, file.filename, status="failed", error=str(exc))
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Upload failed for %s", file.filename)
        metadata_store.upsert_document(doc_id, file.filename, status="failed", error=str(exc))
        return jsonify({"error": "Internal error processing file"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/query", methods=["POST"])
def query():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")
    filters = payload.get("filters") or {}
    doc_id_filter = filters.get("doc_id")
    # session_id is accepted for forward-compat with server-side session
    # tracking, but conversation memory currently lives in the Streamlit
    # frontend's session state.
    _session_id = payload.get("session_id")

    if not question or not question.strip():
        return jsonify({"error": "'question' is required"}), 400

    try:
        query_vector = embed_query(llm_service, question, cache)
        context_chunks = retrieve_context(
            vector_store, metadata_store, query_vector,
            k=Config.TOP_K, doc_id_filter=doc_id_filter,
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        return jsonify({"error": "Retrieval failed", "detail": str(exc)}), 500

    sources = [
        {"text": c["text"], "source": c["source"], "page": c.get("page")}
        for c in context_chunks
    ]
    messages = llm_service.build_messages(question, context_chunks)

    use_streaming = request.args.get("stream", "true").lower() != "false"

    if not use_streaming:
        cache_key = _query_cache_key(question, doc_id_filter)
        cached = cache.get_query(cache_key)
        if cached is not None:
            return jsonify(cached), 200
        try:
            response = llm_service.generate(messages, stream=False)
            answer = response.choices[0].message.content
            result = {"answer": answer, "sources": sources}
            cache.set_query(cache_key, result)
            return jsonify(result), 200
        except Exception as exc:
            logger.exception("Generation failed")
            return jsonify({"error": "Generation failed", "detail": str(exc)}), 500

    def event_stream():
        try:
            stream = llm_service.generate(messages, stream=True)
            for event in stream:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"
        except Exception as exc:
            logger.exception("Generation failed")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    doc = metadata_store.get_document(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    try:
        int_ids = metadata_store.delete_by_doc_id(doc_id)
        if int_ids:
            vector_store.delete(int_ids)
        metadata_store.delete_document(doc_id)
        return jsonify({"status": "deleted", "doc_id": doc_id}), 200
    except Exception:
        logger.exception("Failed to delete document %s", doc_id)
        return jsonify({"error": "Failed to delete document"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT, debug=(Config.FLASK_ENV == "development"))
