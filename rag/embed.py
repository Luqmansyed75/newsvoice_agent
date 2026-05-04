import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_and_chunk_data(file_path="data/news.txt", chunk_size=200, chunk_overlap=20):
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

def embed_documents(chunk_size=200, chunk_overlap=20, persist_directory="chroma_db"):
    chunks = load_and_chunk_data(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    print("Initializing HuggingFace Embeddings (free tier: all-MiniLM-L6-v2)...")
    # This uses a free Hugging Face embedding model locally
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
    # 1. Try different chunk sizes and observe results
    sizes_to_test = [50, 100, 200]
    for size in sizes_to_test:
        print(f"\n=== Testing Chunk Size: {size} ===")
        # We store each test in a different folder so we can compare retrieval later
        embed_documents(chunk_size=size, chunk_overlap=10, persist_directory=f"chroma_db_{size}")
