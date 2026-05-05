import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Fix import pathing depending on how the script is run
try:
    from rag.fetch_news import update_news_file
except ModuleNotFoundError:
    from fetch_news import update_news_file

def load_and_chunk_data(query=None, file_path="data/news.txt", chunk_size=300, chunk_overlap=50):
    # Step 1: Pull the latest news down from the internet!
    update_news_file(query=query)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks with size {chunk_size}.")
    
    return chunks

def embed_documents(query=None, chunk_size=300, chunk_overlap=50, persist_directory="chroma_db_300"):
    # Step 2: Delete the old database so we don't mix old news with new news!
    if os.path.exists(persist_directory):
        print(f"Clearing out old database at {persist_directory}...")
        try:
            shutil.rmtree(persist_directory)
        except Exception as e:
            print(f"Warning: Could not clear old database: {e}")
            
    chunks = load_and_chunk_data(query=query, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    print("Initializing HuggingFace Embeddings (free tier: all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Storing chunks into ChromaDB at {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    print("Done storing into Chroma DB!")
    return vectorstore

if __name__ == "__main__":
    # Test the embedding process
    embed_documents(query=None, chunk_size=300, chunk_overlap=50, persist_directory="chroma_db_300")
