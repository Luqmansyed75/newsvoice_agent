import os
import shutil

# Fix imports depending on how the script is run
try:
    from rag.embed import embed_documents
    from rag.retrieve import retrieve_context
except ModuleNotFoundError:
    from embed import embed_documents
    from retrieve import retrieve_context

def run_experiment():
    """
    Runs an automated experiment testing different chunk sizes and overlaps
    to find the optimal balance of relevance and completeness.
    """
    print("🔥 PRO TIP EXPERIMENT INITIATED 🔥")
    print("Don't guess—experiment! Testing combinations to feel the difference.\n")
    
    chunk_sizes = [200, 300, 500]
    overlaps = [20, 50, 100]
    test_query = "Give me all the latest updates on the stock market and technology."
    
    print(f"Test Query: '{test_query}'\n")
    
    for size in chunk_sizes:
        for overlap in overlaps:
            # We don't want an overlap larger than the chunk itself!
            if overlap >= size:
                continue
                
            dir_name = f"chroma_exp_{size}_{overlap}"
            
            print(f"==================================================")
            print(f"🧪 Testing: Chunk Size = {size} | Overlap = {overlap}")
            print(f"==================================================")
            
            # Step 1: Create the database with these settings
            embed_documents(chunk_size=size, chunk_overlap=overlap, persist_directory=dir_name)
            
            # Step 2: Test the retrieval!
            print("\n🔍 RETRIEVAL RESULTS:")
            try:
                context, docs = retrieve_context(test_query, persist_directory=dir_name)
                
                for i, doc in enumerate(docs):
                    print(f"\n--- Result {i+1} (Length: {len(doc.page_content)} chars) ---")
                    # Print the first 150 characters to see the context snippet
                    snippet = doc.page_content.replace("\n", " ")
                    print(f"{snippet[:200]}...")
                    
                print("\n")
            except Exception as e:
                print(f"Retrieval failed: {e}\n")
                
            # Clean up the experimental database so we don't clutter the hard drive
            try:
                shutil.rmtree(dir_name)
            except:
                pass

if __name__ == "__main__":
    run_experiment()
