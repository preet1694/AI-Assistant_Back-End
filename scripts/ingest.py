"""
One-time script for data ingestion, OPTIMIZED FOR RAG.

This script reads all PDF documents from the 'data/' directory, including the
roll number sheet, splits them into meaningful, context-aware chunks, 
generates embeddings, and then stores them in a FAISS vector store.
"""
import sys
import os

# Add the project root to the Python path if necessary
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/"

def create_vector_db_for_rag():
    """
    Creates and saves a FAISS vector store from documents in the data directory.
    This process is optimized for Retrieval-Augmented Generation (RAG).
    """
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"The '{DATA_PATH}' directory is empty or does not exist. Please add PDF files to it.")
        return

    print("Starting document ingestion process for RAG...")
    
    # Load all PDF files from the data directory
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    
    if not documents:
        print(f"No PDF documents could be loaded from '{DATA_PATH}'. Aborting.")
        return

    print(f"Loaded {len(documents)} document(s).")

    # --- Chunking Strategy: The Key to Effective RAG ---
    # Instead of splitting by line, we split into larger, semantically coherent chunks.
    # This ensures that the retrieved context is meaningful for the LLM.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_splits = text_splitter.split_documents(documents)

    if not all_splits:
        print("Could not split documents into any chunks. Check the document content.")
        return

    print(f"Split documents into {len(all_splits)} text chunks.")
    print("\n--- Example of a Text Chunk ---")
    print(all_splits[0].page_content)
    print("---------------------------------\n")
    
    # --- Embedding Model ---
    # This model converts text chunks into numerical vectors.
    print("Loading embedding model 'sentence-transformers/all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'} # Use 'cuda' for GPU
    )

    # --- FAISS Vector Store Creation ---
    print("Creating FAISS vector store from text chunks...")
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)
        
    db = FAISS.from_documents(all_splits, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    print(f"Vector store created and saved successfully at '{DB_FAISS_PATH}'.")

if __name__ == '__main__':
    # To run this script, execute `python -m scripts.ingest` from the project root.
    create_vector_db_for_rag()
