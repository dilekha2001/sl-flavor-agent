"""
Rebuilds chroma_db from scratch, from the .txt files in ./data, using a
single explicit chunking pass. Run this ONCE to fix the duplicate-chunk
issue found by inspect_chunking.py.

Usage:
    1. Delete the existing chroma_db folder first (back it up if unsure):
         rmdir /s /q chroma_db      (Windows)
         rm -rf chroma_db           (Mac/Linux)
    2. python rebuild_chroma.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

DATA_DIR = "./data"
PERSIST_DIR = "./chroma_db"

# Matches the chunk-length pattern already observed in the existing store
# (~130-500 chars per chunk), so retrieval behaviour stays consistent.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

documents = []
for filename in sorted(os.listdir(DATA_DIR)):
    if not filename.endswith(".txt"):
        continue
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = splitter.split_text(text)
    for chunk in chunks:
        documents.append(Document(page_content=chunk, metadata={"source": filename}))

print(f"Loaded {len(documents)} chunks from "
      f"{len(set(d.metadata['source'] for d in documents))} source files.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)

print(f"Rebuilt vectorstore saved to {PERSIST_DIR}. "
      f"Run inspect_chunking.py again to confirm no duplicates remain.")