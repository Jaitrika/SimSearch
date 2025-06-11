import fitz
import hnswlib
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List

class HNSWVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", dim=384):
        self.model = SentenceTransformer(model_name)
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.chunk_texts: List[str] = []
        self.dim = dim
        self.index_inited = False

    def load_pdf(self, path: str, chunk_size=200) -> List[str]:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks

    def build_index(self, chunks: List[str]):
        self.texts = chunks
        embeddings = self.model.encode(chunks)
        self.index = hnswlib.Index(space='cosine', dim=len(embeddings[0]))
        self.index.init_index(max_elements=len(chunks), ef_construction=200, M=16)
        self.index.add_items(embeddings, list(range(len(chunks))))
        self.index.set_ef(50)


    def query(self, question: str, top_k=3):
        query_vector = self.model.encode(question)
        labels, distances = self.index.knn_query(query_vector, k=top_k)
        results = [(float(1 - distances[0][i]), self.texts[labels[0][i]]) for i in range(top_k)]
        return results

    def save(self, filepath="vector_store_hnsw.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump((self.index, self.texts), f)


    def load(self, filepath="vector_store_hnsw.pkl"):
        with open(filepath, "rb") as f:
            self.index, self.texts = pickle.load(f)

