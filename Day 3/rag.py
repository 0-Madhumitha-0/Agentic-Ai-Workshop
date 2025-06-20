# -*- coding: utf-8 -*-
"""RAG.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1o-ByPm0OrVzBO6IFfMOkATWOQ1spufUs
"""

!pip install PyMuPDF

import fitz  # PyMuPDF
from google.colab import files

# Upload PDF
uploaded = files.upload()
pdf_path = next(iter(uploaded))

# Extract text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

raw_text = extract_text_from_pdf(pdf_path)
print("Sample text:\n", raw_text[:1000])  # Preview first 1000 characters

# Split into chunks
def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

chunks = chunk_text(raw_text)
print(f"Total chunks: {len(chunks)}")
print("\nSample chunk:\n", chunks[0])

!pip install sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Embed chunks
embeddings = embedder.encode(chunks, show_progress_bar=True)
embeddings = np.array(embeddings)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("FAISS index built with", len(embeddings), "vectors")

# Define question
question = "What is the main contribution of the paper?"

# Embed question
q_embedding = embedder.encode([question])

# Search top-k chunks
D, I = index.search(q_embedding, k=3)
retrieved_chunks = [chunks[i] for i in I[0]]

print("Top retrieved chunks:\n")
for i, chunk in enumerate(retrieved_chunks, 1):
    print(f"\nChunk {i}:\n{'-'*40}\n{chunk}")

!pip install transformers

from transformers import pipeline

# Load FLAN-T5 model
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

# Prepare prompt
context = "\n".join(retrieved_chunks)
prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

# Generate answer
result = qa_pipeline(prompt, max_length=200, do_sample=False)[0]['generated_text']
print("Answer:\n", result)

print("Source chunks used for the answer:\n")
for idx, chunk in enumerate(retrieved_chunks, 1):
    print(f"\n[Source {idx}]:\n{chunk}")