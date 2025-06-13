# import fitz  # PyMuPDF
# import numpy as np
# import pickle
# from sentence_transformers import SentenceTransformer
# from typing import List, Tuple

# class SimpleVectorStore:
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)
#         self.embeddings: List[Tuple[str, List[float]]] = []  # (chunk_text, embedding)

#     def _cosine_similarity(self, a, b):
#         a = np.array(a)
#         b = np.array(b)
#         return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#     def load_pdf(self, path: str, chunk_size=200) -> List[str]:
#         doc = fitz.open(path)
#         text = "\n".join(page.get_text() for page in doc)
#         chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
#         return chunks

#     def build_index(self, chunks: List[str]):
#         self.embeddings = [(chunk, self.model.encode(chunk).tolist()) for chunk in chunks]

#     def save(self, filepath="vector_store_brute.pkl"):
#         with open(filepath, "wb") as f:
#             pickle.dump(self.embeddings, f)

#     def load(self, filepath="vector_store_brute.pkl"):
#         with open(filepath, "rb") as f:
#             self.embeddings = pickle.load(f)

#     def query(self, question: str, top_k=3):
#         query_emb = self.model.encode(question)
#         scores = [
#             (self._cosine_similarity(query_emb, emb), chunk)
#             for chunk, emb in self.embeddings
#         ]
#         scores.sort(reverse=True, key=lambda x: x[0])
#         return scores[:top_k]


import fitz  # PyMuPDF
import numpy as np
import pickle
from typing import List, Tuple
from simcse_model.embedding import load_custom_simcse  # <-- load your fine-tuned SimCSE

class SimpleVectorStore:
    def __init__(self):
        self.model = load_custom_simcse()  # now using your SimCSE!
        self.embeddings: List[Tuple[str, List[float]]] = []  # (chunk_text, embedding)

    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def load_pdf(self, path: str, chunk_size=200) -> List[str]:
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks

    def build_index(self, chunks: List[str]):
        embeddings = self.model(chunks)  # batch embed all at once
        self.embeddings = [(chunk, emb.tolist()) for chunk, emb in zip(chunks, embeddings)]

    def save(self, filepath="vector_store_brute.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(self.embeddings, f)

    def load(self, filepath="vector_store_brute.pkl"):
        with open(filepath, "rb") as f:
            self.embeddings = pickle.load(f)

    def query(self, question: str, top_k=3):
        query_emb = self.model(question)[0]  # embed and get first
        scores = [
            (self._cosine_similarity(query_emb, emb), chunk)
            for chunk, emb in self.embeddings
        ]
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:top_k]
