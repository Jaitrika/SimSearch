# 🔍 Lightweight Semantic Search Engine with Vector Index Benchmarking

This project explores different vector indexing strategies (Flat, HNSW, IVF, PQ) for building scalable and accurate semantic search systems. Designed for fast and intelligent document retrieval, with future support for Retrieval-Augmented Generation (RAG) workflows.

---

## 📌 Features

- ✅ Extract & chunk text from PDFs
- ✅ Generate embeddings using `SentenceTransformer`
- ✅ Build persistent vector stores (Flat, HNSW, etc.)
- ✅ Benchmark indexing strategies (precision, latency)
- ✅ Query the store with semantic similarity
- 🚀 Ready for integration with LLMs or RAG pipelines

---

## 🧱 Indexing Backends

| Index Type | Speed | Accuracy | Memory | Use Case |
|------------|-------|----------|--------|----------|
| Flat       | 🐢 Slow  | ✅ High   | 🧠 High   | Small corpora, max accuracy |
| HNSW       | ⚡ Fast  | ✅ High   | 💾 Medium | Real-time, production use |
| IVF        | ⚡ Fast  | ⚠️ Medium | 💾 Low    | Large-scale, approximate search |

---

## 🔧 Setup

```bash
git clone https://github.com/yourusername/vector-search-engine
cd vector-search-engine
pip install -r requirements.txt
````

---

## 🚀 Usage

1. **Convert PDFs to Chunks**

```bash
python src/extract_text.py --input data/sample_docs/example.pdf
```

2. **Generate Embeddings**

```bash
python src/embedder.py --model all-MiniLM-L6-v2
```

3. **Build Vector Index (Flat or HNSW)**

```bash
python src/index_builder.py --type flat
python src/index_builder.py --type hnsw
```

4. **Run Query**

```bash
python src/query_engine.py --query "F1 Grand Prix winner"
```

---

## 📊 Benchmarking

All benchmark results are logged in `results/benchmark.csv`. Compare:

* Query time
* Build time
* Memory usage
* Precision\@k

---

## 📚 Future Work

* [ ] Metadata-based filtering (e.g., year, category)
* [ ] Integrate with OpenAI/LLM for full RAG demo
* [ ] Web-based interface using Streamlit or Flask

---

## 🤓 Author

Made by Jaitrika Reddy 🏎️
Find me on [LinkedIn](https://www.linkedin.com/in/jaitrika-reddy-64aa262ab/)
