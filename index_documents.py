import os
import pickle
from dotenv import load_dotenv
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)
from langchain_community.vectorstores import FAISS 

load_dotenv()

def build_index():
    print("🔍 Loading PDF documents...")
    documents = load_pdf_file(data='data/')
    filtered_docs = filter_to_minimal_docs(documents)

    print("✂️ Splitting documents into text chunks...")
    text_chunks = text_split(filtered_docs)

    print("🧠 Loading embedding model...")
    embeddings = download_hugging_face_embeddings()

    print("📦 Creating FAISS index...")
    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)

    print("💾 Saving FAISS index to 'index.faiss'...")
    vectorstore.save_local("index.faiss")

    with open("docs.pkl", "wb") as f:
        pickle.dump(text_chunks, f)

    print("✅ Indexing complete. Ready to use in Streamlit app.")
