import time
from vector_store.brute_store import SimpleVectorStore 
from vector_store.hnsw_store import HNSWVectorStore

pdf_path = "data\\test_f1.pdf"
queries = [
    "How many cars are allowed to compete in a typical Formula 1 race?",
    "Which tracks that hosted races in the 1950 debut season are still on the current F1 calendar?",
    "What is the weekend schedule for a Grand Prix, including practice and qualifying?",
    "What is the total race distance covered in an F1 Sprint format?",
    "During an F1 Sprint weekend, how does car setup and parc fermé differ from a regular GP?",
    "Which countries will host the most races in this F1 season?",
    "What changes are coming to Formula 1's fuel policy in 2026?",
    "Where are most F1 team headquarters located?",
    "Who regulates Formula 1 and who handles its commercial rights?",
    "In which year did Formula 1 become an official World Championship?",
    "Who was the first Formula 1 World Champion?",
    "Which two drivers have won the most World Championships in Formula 1?",
    "Which team holds the record for most Constructors' Championships?",
    "Who plays Sonny Hayes in the upcoming Formula 1 movie?",
    "Which studio is producing the new Formula 1 movie?"
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
