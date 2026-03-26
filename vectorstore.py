from pathlib import Path
from langsmith import traceable

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_PATH
from loader import load_pdfs, split_documents


@traceable(name="get_vectorstore")
def get_vectorstore():
    """
    Traced — Load or build Chroma vector database.
    LangSmith will show:
        - Whether DB was loaded from disk or built fresh
        - How long embedding + DB creation took
    """

    embeddings = OllamaEmbeddings(model="bge-m3")

    db_exists = Path(CHROMA_PATH).exists() and any(Path(CHROMA_PATH).iterdir())

    if db_exists:
        print("Vector database found. Loading from disk...")

        vectorstore = Chroma(
            persist_directory  = CHROMA_PATH,
            embedding_function = embeddings
        )

        print("Vector database loaded successfully.")
        print("Loading chunks for BM25 keyword search...")

        # load_pdfs and split_documents are also @traceable
        # so they appear as child runs inside get_vectorstore in LangSmith
        docs       = load_pdfs()
        all_chunks = split_documents(docs)

    else:
        

        docs       = load_pdfs()
        all_chunks = split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents         = all_chunks,
            embedding         = embeddings,
            persist_directory = CHROMA_PATH
        )

        print("Vector database created and saved to disk.\n")

    return vectorstore, all_chunks