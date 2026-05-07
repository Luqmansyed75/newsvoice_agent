"""
rag/embed.py — Document loading, chunking, and in-memory vector-store creation.

Improvement: builds the ChromaDB entirely in RAM (ephemeral client) instead
of writing to disk.  Since the agent fetches fresh news for every query,
there is no reason to persist the database — keeping it in memory slashes
I/O latency and removes the need to delete/recreate folders each run.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

try:
    from rag.fetch_news import fetch_live_news, detect_date_range
except ModuleNotFoundError:
    from fetch_news import fetch_live_news, detect_date_range


def load_and_chunk_data(query=None, raw_query=None):
    """Fetch fresh news into memory and split it into RAG chunks."""
    # Detect date range from the raw voice query (before keyword extraction)
    days_back = detect_date_range(raw_query or query or "")
    news_text = fetch_live_news(query=query, days_back=days_back)

    if not news_text:
        news_text = "No news articles found for your specific query."

    # Convert the raw text string into a LangChain Document in-memory
    documents = [Document(page_content=news_text, metadata={"source": "newsapi"})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks (size={settings.CHUNK_SIZE}).")
    return chunks


def embed_documents(
    query=None,
    raw_query=None,
    embeddings: HuggingFaceEmbeddings = None,
):
    """
    Fetch live news, chunk it, and build an IN-MEMORY ChromaDB vectorstore.

    Instead of writing to disk and cleaning up old folders, this creates a
    purely ephemeral vector store that lives only in RAM.  It is returned
    directly to the caller (main.py) which then passes it to the retriever.

    Args:
        query:       Search keywords for NewsAPI.
        raw_query:   The original voice query (with time words intact)
                     so detect_date_range can figure out how far back
                     to search.
        embeddings:  A pre-loaded HuggingFaceEmbeddings instance.
                     Pass one in from main.py so the model is never
                     loaded twice.  If None, one is created here
                     (for backwards compatibility / standalone use).

    Returns:
        vectorstore: An in-memory Chroma vectorstore ready for querying.
    """
    chunks = load_and_chunk_data(query=query, raw_query=raw_query)

    if embeddings is None:
        print(f"Initialising embeddings model '{settings.EMBEDDING_MODEL}'...")
        embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    print(f"Building in-memory vector store with {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        # No persist_directory = purely in-memory (ephemeral)
    )
    print("In-memory vector store ready!")
    return vectorstore


if __name__ == "__main__":
    vs = embed_documents(query=None)
    print(f"Vectorstore created with {vs._collection.count()} documents in memory.")
