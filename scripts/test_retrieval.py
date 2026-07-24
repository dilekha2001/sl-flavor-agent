from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

test_queries = ["spicy chicken curry", "vegetarian dish", "coconut sambol", "dessert", "mild curry"]
for q in test_queries:
    results = vectorstore.similarity_search(q, k=2)
    print(f"\nQuery: {q}")
    for r in results:
        print(f" - {r.metadata.get('source', 'unknown')}: {r.page_content[:100]}...")