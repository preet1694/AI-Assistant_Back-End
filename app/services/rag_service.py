"""
Handles the Retrieval-Augmented Generation (RAG) pipeline.
"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # <<< LATEST LIBRARY
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
import os

DB_FAISS_PATH = 'vectorstore/'
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'

def get_rag_chain():
    """
    Builds and returns the RAG chain. Returns None if the vector store doesn't exist.
    """
    if not os.path.exists(DB_FAISS_PATH):
        print(f"Error: Vector store not found at '{DB_FAISS_PATH}'. Please run the ingestion script.")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    retriever = db.as_retriever(search_kwargs={'k': 35})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=settings.GOOGLE_API_KEY)

    prompt_template = """
    You are a helpful college assistant. Based ONLY on the provided context, please answer the user's question accurately.
    If the information to answer the question is not in the context, you MUST state that you don't have enough information to answer.
    Do not use any prior knowledge.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=['context', 'question']
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain