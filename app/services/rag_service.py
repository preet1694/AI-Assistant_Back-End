"""
Handles the Retrieval-Augmented Generation (RAG) pipeline.

This service loads the vector store and the LLM, and creates a chain
to answer queries based on the ingested documents.
"""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings

# Define constants for paths and model names
DB_FAISS_PATH = 'vectorstore/'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def create_rag_chain():
    """Builds and returns the RAG chain."""
    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # Load the pre-built FAISS vector store
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Create a retriever to fetch relevant documents
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # Select the LLM based on whether a Google API key is provided
    if settings.GOOGLE_API_KEY:
        print("Initializing RAG with Google Gemini model.")
        # --- FIX: Changed model name from "gemini-pro" to a current, valid model ---
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=settings.GOOGLE_API_KEY)
    else:
        print("Initializing RAG with local Ollama Mistral model.")
        llm = Ollama(model="mistral")

    # Define the prompt template to guide the LLM
    prompt_template = """
    You are a helpful college assistant. Based on the provided context, please answer the question.
    If the information is not in the context, state that you don't have enough information to answer.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=['context', 'question']
    )

    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Create a single instance of the RAG chain to be used by the service
rag_chain = create_rag_chain()

def get_rag_response(query: str):
    """Invokes the RAG chain to get a response for a given query."""
    return rag_chain.invoke(query)