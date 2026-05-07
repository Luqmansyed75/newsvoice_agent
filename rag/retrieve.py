"""
rag/retrieve.py — Vector-store querying.

Improvement: accepts a pre-built in-memory vectorstore object directly,
removing the need to re-open a disk-based database.  This pairs with the
ephemeral ChromaDB created in embed.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


def retrieve_context(
    query: str,
    vectorstore: Chroma = None,
    k: int = None,
    embeddings: HuggingFaceEmbeddings = None,
):
    """
    Query ChromaDB and return the most relevant news chunks.

    Args:
        query:        The user's natural-language question.
        vectorstore:  A pre-built in-memory Chroma vectorstore (from
                      embed_documents).  This is the primary path used
                      by main.py.
        k:            Number of chunks to retrieve.
        embeddings:   Pre-loaded HuggingFaceEmbeddings instance.  Only
                      needed for standalone/fallback use when no
                      vectorstore is provided.

    Returns:
        (context_str, docs) — concatenated text and the raw Document list.
    """
    k = k or settings.RETRIEVE_K

    if vectorstore is None:
        # Fallback for standalone use / testing — open from disk
        print("[retrieve] No vectorstore passed in — falling back to disk.")
        if embeddings is None:
            print(f"Initialising embeddings model '{settings.EMBEDDING_MODEL}'...")
            embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=settings.CHROMA_DIR,
            embedding_function=embeddings,
        )

    docs    = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    return context, docs


if __name__ == "__main__":
    query = "Give me today's AI news?"
    print(f"QUERY: '{query}'\n")
    context, docs = retrieve_context(query)
    for i, doc in enumerate(docs):
        print(f"Result {i + 1}: {doc.page_content}\n")
