"""src/ui.py — Streamlit chat UI."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Knowledge Assistant", page_icon="🧠")
st.title("🧠 Knowledge Assistant")

# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render existing messages ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption(f"Sources: {', '.join(msg['sources'])}")

# ── Chat input ─────────────────────────────────────────────────────────────────
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
