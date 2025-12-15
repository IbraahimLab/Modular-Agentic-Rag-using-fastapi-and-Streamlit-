from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def build_vectorstore(
    documents,
    embedder,
    persist_dir: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    splits = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        splits,
        embedder,
        persist_directory=persist_dir,
    )
    vectordb.persist()

    return vectordb
