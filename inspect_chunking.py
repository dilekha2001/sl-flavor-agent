"""
Run this once against your existing chroma_db to find out whether your
corpus was chunked or embedded as whole files. Takes seconds, no rebuild.

Usage:
    python inspect_chunking.py
"""
from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

data = vectorstore.get()  # pulls all stored documents + metadata
docs = data["documents"]
metas = data["metadatas"]

print(f"Total chunks/documents stored: {len(docs)}\n")

# Group by source file to see how many chunks exist per original file
from collections import defaultdict
by_source = defaultdict(list)
for doc, meta in zip(docs, metas):
    source = meta.get("source", "unknown") if meta else "unknown"
    by_source[source] = by_source.get(source, [])
    by_source[source].append(len(doc))

print(f"Distinct source files found: {len(by_source)}\n")

multi_chunk_files = {k: v for k, v in by_source.items() if len(v) > 1}

if multi_chunk_files:
    print("=> CHUNKING WAS USED. These files were split into multiple pieces:")
    for source, lengths in list(multi_chunk_files.items())[:5]:
        print(f"   {source}: {len(lengths)} chunks, char lengths = {lengths}")
    print("\nLook at the character lengths above to estimate your chunk_size.")
else:
    print("=> NO CHUNKING. Each source file = exactly one document/embedding.")
    sample_lengths = [v[0] for v in list(by_source.values())[:5]]
    print(f"   Sample whole-file lengths (chars): {sample_lengths}")