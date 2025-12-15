import sys
from pathlib import Path
import streamlit as st

# --- make project root importable ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from langchain_core.messages import HumanMessage, AIMessage

from server.config import CONFIG
from server.rag.loaders import load_pdf
from server.rag.embeddings import get_embedder
from server.rag.vectorstore import build_vectorstore
from server.agent.tools import build_tools
from server.agent.graph import build_agent
from shared.utils import file_sha256

# -----------------------------
# Streamlit page config
# -----------------------------
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
if "chat" not in st.session_state:
    st.session_state.chat = []

if "agent" not in st.session_state:
    st.session_state.agent = None

# -----------------------------
# PDF upload & indexing
# -----------------------------
uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded and st.session_state.agent is None:
    workspace = Path(CONFIG["WORKSPACE_DIR"])
    workspace.mkdir(exist_ok=True)

    file_bytes = uploaded.getvalue()
    file_id = file_sha256(file_bytes)
    pdf_path = workspace / f"{file_id}.pdf"
    pdf_path.write_bytes(file_bytes)

    with st.spinner("Indexing PDF…"):
        docs = load_pdf(str(pdf_path))
        embedder = get_embedder(CONFIG["EMBEDDING_MODEL"])

        vectordb = build_vectorstore(
            documents=docs,
            embedder=embedder,
            persist_dir=str(workspace / f"chroma_{file_id}"),
        )

        retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        tools = build_tools(retriever, CONFIG["SERPER_API_KEY"])
        st.session_state.agent = build_agent(
            CONFIG["MODEL_NAME"],
            CONFIG["GROQ_API_KEY"],
            tools,
        )

    st.success("PDF indexed successfully")

# -----------------------------
# Render full conversation
# -----------------------------
for msg in st.session_state.chat:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# -----------------------------
# Chat input
# -----------------------------
prompt = st.chat_input("Ask a question about the PDF")

if prompt and st.session_state.agent:
    # user message
    user_msg = HumanMessage(prompt)
    st.session_state.chat.append(user_msg)

    with st.chat_message("user"):
        st.write(prompt)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = st.session_state.agent.invoke(
                {"messages": st.session_state.chat}
            )
            st.session_state.chat = result["messages"]

            last_ai = next(
                m for m in reversed(st.session_state.chat)
                if isinstance(m, AIMessage)
            )
            st.write(last_ai.content)