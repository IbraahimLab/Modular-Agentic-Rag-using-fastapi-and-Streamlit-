from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from typing import List
import uuid

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from server.config import CONFIG
from server.rag.loaders import load_pdf
from server.rag.embeddings import get_embedder
from server.rag.vectorstore import build_vectorstore
from server.agent.tools import build_tools
from server.agent.graph import build_agent
from shared.utils import file_sha256
from server.observability.langsmith import init_langsmith
init_langsmith()


app = FastAPI(title="Agentic RAG API")

# -----------------------------
# In-memory session store
# (replace with Redis later)
# -----------------------------
SESSIONS = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Upload PDF
# -----------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())

    workspace = Path(CONFIG["WORKSPACE_DIR"])
    workspace.mkdir(exist_ok=True)

    file_bytes = await file.read()
    file_id = file_sha256(file_bytes)

    pdf_path = workspace / f"{file_id}.pdf"
    pdf_path.write_bytes(file_bytes)

    docs = load_pdf(str(pdf_path))
    embedder = get_embedder(CONFIG["EMBEDDING_MODEL"])

    vectordb = build_vectorstore(
        documents=docs,
        embedder=embedder,
        persist_dir=str(workspace / f"chroma_{file_id}"),
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    tools = build_tools(retriever, CONFIG["SERPER_API_KEY"])

    agent = build_agent(
        CONFIG["MODEL_NAME"],
        CONFIG["GROQ_API_KEY"],
        tools,
    )

    SESSIONS[session_id] = {
        "agent": agent,
        "messages": [],
    }

    return {"session_id": session_id}

# -----------------------------
# Chat
# -----------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = SESSIONS.get(req.session_id)

    if not session:
        return {"answer": "Invalid session. Please upload a PDF again."}

    messages: List[BaseMessage] = session["messages"]
    messages.append(HumanMessage(req.message))

    result = session["agent"].invoke({"messages": messages})
    session["messages"] = result["messages"]

    last_ai = next(
        m for m in reversed(session["messages"])
        if isinstance(m, AIMessage)
    )

    return {"answer": last_ai.content}
