import time
from brute_store import SimpleVectorStore 
from hnsw_store import HNSWVectorStore

pdf_path = "data/mobilllama-1-6.pdf"
queries = [
    "What is the architecture of MobilLLaMA?",
    "How much memory does the model use?",
    "What's the tokenizer used?"
]

def benchmark(store_class, label):
    print(f"\n--- {label} ---")
    store = store_class()
    chunks = store.load_pdf(pdf_path)
    store.build_index(chunks)

    for q in queries:
        start = time.time()
        results = store.query(q)
        end = time.time()
        print(f"\nQuery: {q}")
        print(f"Time: {(end - start):.4f}s")
        print(f"Top Result: {results[0][0]:.4f} - {results[0][1][:150]}...")

benchmark(SimpleVectorStore, "Brute-Force Search")
benchmark(HNSWVectorStore, "HNSW Search")
