"""src/ui.py — Streamlit chat UI."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Personal Assistant RAG", page_icon="🧠", layout="wide")

# ── Sidebar — file upload + namespace selector ─────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    namespace = st.text_input("Namespace", value="default",
                              help="Logical partition — use a unique name per person or topic")

    st.divider()
    st.header("📂 Ingest a Document")
    uploaded = st.file_uploader(
        "Upload any file",
        type=["pdf", "txt", "docx", "csv", "json", "xlsx",
              "jpg", "jpeg", "png", "mp3", "wav", "m4a", "vtt", "srt"],
        help="File is sent to /ingest, embedded, and stored in Pinecone under your namespace",
    )
    if uploaded and st.button("Ingest →"):
        with st.spinner(f"Ingesting {uploaded.name}…"):
            try:
                resp = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    data={"namespace": namespace},
                    timeout=120,
                )
                result = resp.json()
                if resp.status_code == 200:
                    st.success(
                        f"✅ **{result['filename']}**\n\n"
                        f"{result['chunks_upserted']} new chunks stored in `{result['namespace']}`"
                    )
                else:
                    st.error(f"Error: {result.get('detail', resp.text)}")
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")

# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("🧠 Personal Assistant RAG")

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

# Chat input
if prompt := st.chat_input("Ask me anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking…"):
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                    "namespace": namespace,
                },
                timeout=60,
            ).json()
        except requests.RequestException as exc:
            st.error(f"API error: {exc}")
            st.stop()

    st.session_state.session_id = resp["session_id"]
    answer = resp["answer"]
    sources = resp.get("sources", [])

    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
