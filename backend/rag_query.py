# rag_query.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import pipeline

# -------------------------------
# Step 1: Load Embedding Model
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# Step 2: Load Vector Index and Docs
# -------------------------------
index = faiss.read_index("faiss_index.index")

with open("documents.pkl", "rb") as f:
    documents = pickle.load(f)

# -------------------------------
# Step 3: Load Hugging Face LLM
# -------------------------------
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# -------------------------------
# Step 4: Get Relevant Chunks
# -------------------------------
def get_top_k_docs(query, k=3):
    query_embedding = model.encode([query])
    scores, doc_ids = index.search(np.array(query_embedding), k)
    return [documents[i] for i in doc_ids[0]]

# -------------------------------
# Step 5: Generate Answer
# -------------------------------
def generate_answer(query):
    top_docs = get_top_k_docs(query)
    context = "\n".join(top_docs)

    prompt = f"""Use the following GLA University information to answer the question:
Context: {context}
Question: {query}"""

    response = generator(prompt, max_length=256, do_sample=True)[0]['generated_text']
    return response

# -------------------------------
# Step 6: Interactive CLI
# -------------------------------
if __name__ == "__main__":
    query = input("Ask your GLA-related question: ")
    answer = generate_answer(query)
    print("\nAnswer:", answer)
