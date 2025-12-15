import streamlit as st
import requests

# -----------------------------
# Config
# -----------------------------
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Agentic RAG Chat")
st.caption("Upload a PDF and ask questions about it.")

# -----------------------------
# Session state
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# PDF upload
# -----------------------------
uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded and st.session_state.session_id is None:
    with st.spinner("Uploading and indexing PDF…"):
        response = requests.post(
            f"{API_BASE}/upload_pdf",
            files={"file": uploaded.getvalue()},
        )

    if response.status_code == 200:
        st.session_state.session_id = response.json()["session_id"]
        st.success("PDF indexed successfully")
    else:
        st.error("Failed to upload PDF")

# -----------------------------
# Render conversation
# -----------------------------
for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.write(content)

# -----------------------------
# Chat input
# -----------------------------
prompt = st.chat_input("Ask a question about the PDF")

if prompt and st.session_state.session_id:
    # show user message
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)

    # call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = requests.post(
                f"{API_BASE}/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                },
            )

            if response.status_code == 200:
                answer = response.json()["answer"]
            else:
                answer = "Server error. Please try again."

            st.session_state.chat.append(("assistant", answer))
            st.write(answer)
