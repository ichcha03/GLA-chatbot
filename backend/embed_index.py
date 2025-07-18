# embed_index.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load text data
with open("data/gla_data.txt", "r", encoding='utf-8') as f:
    documents = [line.strip() for line in f if line.strip()]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(documents)

# Save embeddings and documents
with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

# Create and save FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))
faiss.write_index(index, "faiss_index.index")

print("Index and documents saved successfully.")
