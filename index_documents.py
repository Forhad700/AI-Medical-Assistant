import os
import pickle
from dotenv import load_dotenv
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)
from langchain.vectorstores import FAISS

# Load environment variables (if needed)
load_dotenv()

# Step 1: Load PDF documents
print("🔍 Loading PDF documents...")
documents = load_pdf_file(data='data/')
filtered_docs = filter_to_minimal_docs(documents)

# Step 2: Split into chunks
print("✂️ Splitting documents into text chunks...")
text_chunks = text_split(filtered_docs)

# Step 3: Load embeddings
print("🧠 Loading embedding model...")
embeddings = download_hugging_face_embeddings()

# Step 4: Create FAISS vector store
print("📦 Creating FAISS index...")
vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)

# Step 5: Save FAISS index and documents
print("💾 Saving FAISS index to 'index.faiss'...")
vectorstore.save_local("index.faiss")

with open("docs.pkl", "wb") as f:
    pickle.dump(text_chunks, f)

print("✅ Indexing complete. Ready to use in Streamlit app.")
