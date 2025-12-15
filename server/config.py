# =============================
# CONFIG
# =============================
from dotenv import load_dotenv
import os
import streamlit as st

def load_config():
    load_dotenv()

    config = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        "MODEL_NAME": "moonshotai/kimi-k2-instruct-0905",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "WORKSPACE_DIR": ".rag_workspace",
    }

    if not config["GROQ_API_KEY"]:
        st.error("❌ GROQ_API_KEY missing in environment.")
    if not config["SERPER_API_KEY"]:
        st.error("❌ SERPER_API_KEY missing in environment.")

    return config


CONFIG = load_config()
