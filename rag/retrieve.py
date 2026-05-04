from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def retrieve_context(query, persist_directory="chroma_db_200", k=2):
    """
    Query the Vector DB and retrieve relevant news chunks.
    """
    # Re-initialize embeddings to query the DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load existing Chroma DB
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )
    
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    return context, docs

if __name__ == "__main__":
    # 2. Observe results by testing retrieval against the different chunk sizes we stored
    query = "Give me today's AI news?"
    print(f"QUERY: '{query}'\n")
    
    for size in [50, 100, 200]:
        print(f"--- Retrieving from DB with chunk size {size} ---")
        try:
            context, docs = retrieve_context(query, persist_directory=f"chroma_db_{size}")
            for i, doc in enumerate(docs):
                print(f"Result {i+1}: {doc.page_content}")
            print("\n")
        except Exception as e:
            print(f"Error retrieving from chroma_db_{size}: {e}")
