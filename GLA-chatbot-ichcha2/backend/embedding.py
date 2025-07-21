from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


from langchain.schema import Document

import json


# Load data
with open(r"D:\gla2 chatbot\GLA-chatbot-ichcha2\cleaned_data\part1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare documents
docs = []
for item in data:
    docs.append(Document(
        page_content=item["question"] + " " + item["answer"],
        metadata={"question": item["question"]}
    ))

# Use Sentence Transformer for Embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create ChromaDB
db = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./gla_chroma"
)
db.persist()
print("✅ Vector store created and persisted!")
