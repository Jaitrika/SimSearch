from vector_store.brute_store import SimpleVectorStore 
from vector_store.hnsw_store import HNSWVectorStore

import os

pdf_path = "data\\ai_paragraph.pdf"

# Choose the vector store
use_hnsw = input("Use HNSW? (y/n): ").lower() == 'y'

# Set the correct file and class based on choice
store_file = "vector_store_hnsw.pkl" if use_hnsw else "vector_store_brute.pkl"
store = HNSWVectorStore() if use_hnsw else SimpleVectorStore()

# Load or build
if os.path.exists(store_file):
    print("🔁 Loading saved vector store...")
    store.load(store_file)
else:
    print("📄 Reading and embedding PDF...")
    chunks = store.load_pdf(pdf_path)
    store.build_index(chunks)
    store.save(store_file)
    print("✅ Saved vector store to disk.")

# Query loop
while True:
    question = input("\nAsk a question (or 'exit'): ")
    if question.lower() == "exit":
        break
    results = store.query(question)
    print("\nTop results:")
    for score, chunk in results:
        print(f"\n[Score: {score:.4f}]\n{chunk.strip()[:300]}")
