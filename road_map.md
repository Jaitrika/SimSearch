# 🧭 Project Roadmap: SimCSE from Scratch + Vector Search System

## ✅ Phase 0: Setup (Done / Almost Done)
- [x] Custom vector store with semantic search ✅
- [x] Ability to load and chunk PDFs to sentence-level ✅
- [x] Store + reuse embeddings for future runs ✅

---

## 🔥 Phase 1: Build SimCSE from Scratch
> Goal: Train a sentence-level embedding model using contrastive learning.

### 🧪 1A. Base SimCSE (Unsupervised)
- [ ] Load a transformer encoder (MiniLM / DistilBERT)
- [ ] Implement unsupervised SimCSE (dropout noise)
- [ ] Use a large corpus (Wikipedia, StackExchange, Common Crawl)
- [ ] Train with InfoNCE loss (or NT-Xent)
- [ ] Save model checkpoint

### 🎯 1B. Supervised SimCSE
- [ ] Use SNLI or other NLI datasets
- [ ] Implement supervised contrastive learning (entailment = pos, contradiction = neg)
- [ ] Add hard negative mining logic
- [ ] Save separate supervised model

---

## 📊 Phase 2: Evaluation + Benchmarking

> Goal: Show your model actually works well and how it compares.

### 📏 2A. STS Evaluation
- [ ] Load STS-B, QuoraQP, or SICK dataset
- [ ] Embed sentence pairs → cosine sim → Spearman correlation
- [ ] Compare with `all-MiniLM` baseline

### 🔍 2B. Retrieval Evaluation
- [ ] Create a mini QA dataset
- [ ] Embed questions and answers → do top-k search
- [ ] Evaluate precision@k

### 🌈 2C. Visualization
- [ ] t-SNE or UMAP plot of embeddings
- [ ] Show clustering of similar sentences
- [ ] Color by label (if supervised)

---

## ⚙️ Phase 3: Integration Into Vector Store

> Plug your trained model into your previously built system.

- [ ] Replace embedding backend with your SimCSE model
- [ ] Save/load vector store to disk
- [ ] Benchmark speed and accuracy vs MiniLM or `bge-small-en`

---

## 🚀 Phase 4: Indexing & Optimization

> Engineer-level work to scale vector search.

- [ ] Add HNSWlib indexing
- [ ] Add Faiss indexing (Flat, IVF, PQ)
- [ ] Measure latency, memory, recall@k for each
- [ ] Add metadata filtering support (chunk id, source page)

---

## 🖥️ Phase 5: Demo App (Optional but ✨)

> Add UX to showcase what you’ve built.

- [ ] Streamlit or Flask app
- [ ] Upload PDF → chunk → embed → search
- [ ] Choose model from dropdown: SimCSE (unsup), SimCSE (sup), MiniLM
- [ ] Show t-SNE live preview

---

## 📚 Phase 6: Polish & Publish

> The final touches to make it stand out.

- [ ] Add training logs + learning curve plots
- [ ] Write a detailed README with:
  - Motivation
  - Dataset
  - Model architecture
  - Training process
  - Evaluation results
  - Demo video/gif
- [ ] Publish on GitHub + HuggingFace 🤝
- [ ] Deploy demo on HuggingFace Spaces or Streamlit Cloud

---

# 🧠 Bonus Ideas (Stretch Goals)
- [ ] Use LoRA / QLoRA to finetune base model with fewer resources
- [ ] Train a distilled version of your model
- [ ] ONNX export for inference speed
- [ ] Multilingual version (Indic, etc.)

---

🧪 Total Effort: ~3-4 weeks (intense) or ~6-8 weeks (moderate pace)  
📍 Outcome: A research-grade project that blends theory, engineering, and application.

