from pathlib import Path
from langsmith import traceable

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import PDF_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP


@traceable(name="load_pdfs")
def load_pdfs():
    """
    Traced — Load all PDF files from PDF_FOLDER.
    LangSmith will show how many pages were loaded and how long it took.
    """

    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in '{PDF_FOLDER}' folder. "
            "Please add your PDF files there and try again."
        )

    print(f"\nFound {len(pdf_files)} PDF file(s). Loading...")

    all_docs = []
    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        docs   = loader.load()
        all_docs.extend(docs)
        print(f"  Loaded: {pdf_path.name}  ({len(docs)} pages)")

    print(f"Total pages loaded: {len(all_docs)}\n")
    return all_docs


@traceable(name="split_documents")
def split_documents(docs):
    """
    Traced — Split pages into smaller chunks.
    LangSmith will show how many chunks were created and how long it took.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}\n")
    return chunks