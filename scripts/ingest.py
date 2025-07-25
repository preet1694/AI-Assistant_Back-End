# File: scripts/ingest.py
"""
One-time script for data ingestion.

This script reads PDF documents from the 'data/' directory, processes the text
line-by-line to create clean, self-contained records, and then stores them
in a FAISS vector store.
"""
import sys
import os
import re

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define constants for paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/"

def create_vector_db():
    """Creates and saves a FAISS vector store from documents in the data directory."""
    print("Starting document ingestion process...")
    
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    
    if not documents:
        print(f"No PDF documents found in '{DATA_PATH}'. Aborting.")
        return

    # --- FINAL, MOST ROBUST FIX: Process text line-by-line and clean it ---
    all_records = []
    for doc in documents:
        full_text = doc.page_content
        # Split by newline and filter out empty lines
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        
        for line in lines:
            # Use a regular expression to find lines that look like student records
            # This is more reliable than just checking for "IT"
            if re.match(r'^IT\d{3}', line):
                # Create a clean LangChain Document for each valid record
                all_records.append(Document(
                    page_content=line, 
                    metadata={"source": doc.metadata.get('source', 'unknown')}
                ))

    if not all_records:
        print("Could not extract any valid student records from the documents.")
        return

    print(f"Processed document into {len(all_records)} individual line-based records.")
    print("--- Example Record ---")
    print(all_records[0].page_content)
    print("----------------------")
    
    print(f"Loading embedding model 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

    print("Creating FAISS vector store from cleaned line-based records...")
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)
        
    db = FAISS.from_documents(all_records, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    print(f"Vector store created and saved successfully at '{DB_FAISS_PATH}'.")

if __name__ == '__main__':
    create_vector_db()
