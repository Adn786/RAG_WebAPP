"""Streamlit chat UI for the RAG application. Talks to the Flask API over HTTP
(SSE for streamed answers) and keeps conversation history in session state."""
import json
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
MAX_HISTORY = 20  # number of chat messages retained in session state

st.set_page_config(page_title="RAG Chat", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str, "sources": [...]}]
if "documents" not in st.session_state:
    st.session_state.documents = []


def refresh_documents():
    try:
        resp = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        if resp.ok:
            st.session_state.documents = resp.json().get("documents", [])
    except requests.RequestException:
        pass


def upload_file(file):
    files = {"file": (file.name, file.getvalue())}
    return requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)


def stream_query(question, session_id, doc_id_filter=None):
    payload = {
        "question": question,
        "session_id": session_id,
        "filters": {"doc_id": doc_id_filter} if doc_id_filter else {},
    }
    with requests.post(f"{BACKEND_URL}/query", json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            yield json.loads(line[len("data:"):].strip())


def render_sources(sources):
    with st.expander("Sources"):
        for j, s in enumerate(sources, start=1):
            page = f", page {s['page']}" if s.get("page") else ""
            st.markdown(f"**[{j}] {s['source']}{page}**")
            st.caption(s["text"][:500])


# --- Sidebar: upload + document filter -----------------------------------
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload a document", type=["pdf", "docx", "pptx", "txt", "xlsx", "csv"]
    )
    if uploaded is not None and st.button("Process file"):
        with st.spinner(f"Processing {uploaded.name}..."):
            try:
                resp = upload_file(uploaded)
                if resp.ok:
                    data = resp.json()
                    st.success(f"Indexed {data['num_chunks']} chunks from {uploaded.name}")
                    refresh_documents()
                else:
                    st.error(resp.json().get("error", "Upload failed"))
            except requests.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")

    refresh_documents()
    doc_options = {"All documents": None}
    for d in st.session_state.documents:
        if d["status"] == "ready":
            doc_options[f"{d['filename']} ({d['num_chunks']} chunks)"] = d["doc_id"]

    selected_label = st.selectbox("Filter by document", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_label]

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Main: chat interface --------------------------------------------------
st.title("RAG Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Thinking..._")
        full_text = ""
        sources = []
        try:
            for event in stream_query(question, session_id="default", doc_id_filter=selected_doc_id):
                if "token" in event:
                    full_text += event["token"]
                    placeholder.markdown(full_text + "▌")
                elif "done" in event:
                    sources = event.get("sources", [])
                elif "error" in event:
                    full_text += f"\n\n*Error: {event['error']}*"
            placeholder.markdown(full_text or "_No response_")
            if sources:
                render_sources(sources)
        except requests.RequestException as exc:
            full_text = f"Could not reach backend: {exc}"
            placeholder.markdown(full_text)

    st.session_state.messages.append({"role": "assistant", "content": full_text, "sources": sources})
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
