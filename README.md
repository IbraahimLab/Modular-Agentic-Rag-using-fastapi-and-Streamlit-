# Agentic RAG with FastAPI and Streamlit

A lightweight end-to-end Retrieval-Augmented Generation (RAG) demo. It lets you upload a PDF, indexes it into a vector store, and chats with an agent that can search the PDF, the web, and arXiv. The backend is built with FastAPI + LangGraph; the frontend is a simple Streamlit chat UI.

---
## Project layout

```
.
├── client/          # Streamlit chat UI
├── server/          # FastAPI app, LangGraph agent, RAG stack, observability helpers
├── shared/          # Small shared utilities
├── evaluation/      # RAGAS evaluation runner
├── pyproject.toml   # Python dependencies
└── README.md
```

### Key pieces
- **`server/main.py`**: FastAPI routes to upload a PDF and chat. Creates sessions, builds the RAG stack, and stores the LangGraph agent per session.
- **`server/rag/*`**: PDF loader, HuggingFace embeddings, Chroma vector store builder.
- **`server/agent/*`**: Tooling (PDF search, web search via Serper, arXiv search) and LangGraph graph definition.
- **`client/app.py`**: Streamlit chat surface that uploads the PDF, then sends chat turns to the API.
- **`evaluation/run_ragas.py`**: Skeleton for running RAGAS evaluations on stored conversations.
- **`shared/utils.py`**: Simple helpers (e.g., PDF file hashing).

---
## Prerequisites
- Python 3.11+
- `uv` or `pip` for installing dependencies
- Access tokens:
  - `GROQ_API_KEY` for the LLM
  - `SERPER_API_KEY` for Google Serper search
- PDF with extractable text (PyPDF is used for parsing)

---
## Installation

1. (Recommended) create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies (using `uv` or `pip`):
   ```bash
   uv pip install -r <(uv pip compile pyproject.toml)
   ```
   or
   ```bash
   pip install -e .
   ```

3. Create a `.env` file in the project root with your keys:
   ```bash
   GROQ_API_KEY=your_groq_key
   SERPER_API_KEY=your_serper_key
   ```

---
## Running the stack

### 1) Start the API
```bash
uvicorn server.main:app --reload --port 8000
```
What it does on first PDF upload:
- Saves the PDF to `.rag_workspace/<hash>.pdf`
- Splits & embeds text with `sentence-transformers/all-MiniLM-L6-v2`
- Stores vectors in a persisted Chroma DB
- Builds tools: PDF search, Serper web search, arXiv search
- Spins up a LangGraph agent bound to those tools

### 2) Launch the Streamlit client
In a separate terminal:
```bash
streamlit run client/app.py --server.port 8501
```
Then open http://localhost:8501.

---
## How chatting works
1. Upload a PDF via the Streamlit UI.
2. The backend indexes the PDF and creates a session-specific agent.
3. Each question is sent to `/chat` along with the session ID.
4. The LangGraph agent decides whether to call tools (PDF search, web search, arXiv) before replying.
5. The latest assistant message is returned to the UI and shown in the chat history.

---
## API quick reference
- `GET /health` → `{ "status": "ok" }`
- `POST /upload_pdf` (multipart `file`) → `{ "session_id": "..." }`
- `POST /chat` (JSON `{ session_id, message }`) → `{ "answer": "..." }`

---
## Configuration
`server/config.py` loads settings from environment variables and defaults:
- `GROQ_API_KEY` (required)
- `SERPER_API_KEY` (required)
- `MODEL_NAME` (default: `moonshotai/kimi-k2-instruct-0905`)
- `EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `WORKSPACE_DIR` (default: `.rag_workspace`)

If keys are missing, errors are surfaced in Streamlit to help you catch setup issues early.

---
## Evaluation (optional)
`evaluation/run_ragas.py` is a starter script for running RAGAS evaluations against saved conversations. Customize it with your dataset and metrics to benchmark answer quality.

---
## Troubleshooting tips
- Make sure `GROQ_API_KEY` and `SERPER_API_KEY` are set before starting the API.
- Remove `.rag_workspace` if you want to clear cached PDFs and vector stores.
- If a PDF has no extractable text, PyPDF will raise an error during upload.
- For cross-origin setups, adjust `API_BASE` in `client/app.py` to point to your API host.

---
## Contributing
Feel free to open issues or PRs that improve reliability, add new tools, or enhance the UI. Keep code simple and documented so the project stays approachable.

---
## License
MIT (see repository license if provided).
