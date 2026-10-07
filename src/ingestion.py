import os
import glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

def extract_metadata(file_path: Path):
    name = file_path.stem.lower()
    if "nvidia" in name:
        return {"company": "NVIDIA", "ticker": "NVDA", "quarter": "Q3", "fiscal_year": "2025"}
    elif "apple" in name:
        return {"company": "Apple", "ticker": "AAPL", "quarter": "Q4", "fiscal_year": "2024"}
    elif "microsoft" in name:
        return {"company": "Microsoft", "ticker": "MSFT", "quarter": "Q1", "fiscal_year": "2025"}
    return {"company": "Unknown", "ticker": "N/A", "quarter": "N/A", "fiscal_year": "N/A"}

def ingest_documents(chunk_size: int = 600, chunk_overlap: int = 100, persist_dir: Path = CHROMA_DIR):
    print(f"[*] Starting financial corpus ingestion (ChunkSize={chunk_size}, Overlap={chunk_overlap})...")
    files = list(RAW_DOCS_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No financial transcripts found in {RAW_DOCS_DIR}")

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for f in files:
        loader = TextLoader(str(f), encoding="utf-8")
        docs = loader.load()
        meta = extract_metadata(f)
        for doc in docs:
            doc.metadata.update(meta)
            doc.metadata["source_file"] = f.name
            splits = splitter.split_documents([doc])
            all_docs.extend(splits)

    print(f"[+] Loaded {len(files)} files, produced {len(all_docs)} chunks.")

    # Using HuggingFace local embeddings (all-MiniLM-L6-v2) for robust local indexing
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize / overwrite persistent ChromaDB
    vector_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="finsecure_corpus"
    )
    print(f"[+] Successfully indexed {len(all_docs)} chunks into ChromaDB at {persist_dir}")
    return vector_store

if __name__ == "__main__":
    ingest_documents()
