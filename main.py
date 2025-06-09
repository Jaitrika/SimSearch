from vector_store import SimpleVectorStore
import os

pdf_path = "data\mobilllama-1-6.pdf"
store_file = "vector_store.pkl"

store = SimpleVectorStore()

if os.path.exists(store_file):
    print("🔁 Loading saved vector store...")
    store.load(store_file)
else:
    print("📄 Reading and embedding PDF...")
    chunks = store.load_pdf(pdf_path)
    store.build_index(chunks)
    store.save(store_file)
    print("✅ Saved vector store to disk.")

# Example query
while True:
    question = input("\nAsk a question (or 'exit'): ")
    if question.lower() == "exit":
        break
    results = store.query(question)
    print("\nTop results:")
    for score, chunk in results:
        print(f"\n[Score: {score:.4f}]\n{chunk.strip()[:300]}")
