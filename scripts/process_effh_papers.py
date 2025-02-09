import os
import glob
import faiss
import chromadb
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from llama_index import SimpleDirectoryReader, VectorStoreIndex

# Set OpenAI API Key (or use a local model)
openai.api_key = "your-api-key"  # Replace with your OpenAI API Key or remove for local models

# Directory containing research papers
DATA_DIR = "data/"

# Load research papers
def load_research_papers(data_dir):
    reader = SimpleDirectoryReader(data_dir)
    docs = reader.load_data()
    return docs

# Process text into smaller chunks for AI retrieval
def split_text_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc.text))
    return chunks

# Create vector embeddings using OpenAI or local transformer model
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()  # Uses OpenAI Embeddings
    # Alternative: Use local models like sentence-transformers
    return embeddings, FAISS.from_texts(chunks, embeddings)

# Store embeddings in FAISS (or ChromaDB)
def store_embeddings(vector_db, db_path="faiss_index"):
    vector_db.save_local(db_path)

# Retrieve answers based on user queries
def retrieve_answer(vector_db, query):
    docs = vector_db.similarity_search(query, k=3)  # Retrieve top 3 most relevant chunks
    return "\n".join([doc.page_content for doc in docs])

# Main execution
if __name__ == "__main__":
    print("Loading research papers...")
    documents = load_research_papers(DATA_DIR)

    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(documents)

    print("Creating embeddings...")
    embeddings, vector_db = create_embeddings(chunks)

    print("Storing embeddings in FAISS...")
    store_embeddings(vector_db)

    print("AI system ready! You can now query the AI about EFFH.")

    # Example query:
    query = "What does the Entropic Force-Field Hypothesis say about gravity?"
    print("\nAI Response:\n", retrieve_answer(vector_db, query))
