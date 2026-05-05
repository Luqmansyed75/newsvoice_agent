import os
from huggingface_hub import InferenceClient

def extract_keywords(query):
    """
    Uses the HuggingFace LLM to extract the core search keywords from a natural language query.
    """
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token or hf_token == "your_hugging_face_token_here":
        return query # Fallback
        
    client = InferenceClient(api_key=hf_token)
    
    prompt_text = f"Extract the main 2-3 search keywords from the following question. Output ONLY the keywords separated by spaces. Do not output any conversational text or punctuation.\nQuestion: \"{query}\"\nKeywords:"
    
    messages = [{"role": "user", "content": prompt_text}]
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            max_tokens=15,
            temperature=0.1
        )
        return response.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"Keyword extraction failed: {e}")
        return query

def build_prompt_and_summarize(query, context, mode="standard"):
    """
    Combines the user's query and the retrieved context, then sends it to 
    a HuggingFace free-tier model for summarization using the modern InferenceClient.
    """
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token or hf_token == "your_hugging_face_token_here":
        print("WARNING: HUGGINGFACEHUB_API_TOKEN is not set properly in your .env file.")
        return "Error: Missing Hugging Face Token."

    print("Connecting to Hugging Face Inference API...")
    client = InferenceClient(api_key=hf_token)
    
    # ─── New Feature: Dynamic Prompting based on Mode ───
    if mode == "story":
        system_instruction = "You are a master storyteller. Use the provided news context to answer the user's query. Weave the facts into a captivating, dramatic, and entertaining narrative."
    elif mode == "1min":
        system_instruction = "You are a fast-paced news anchor. Summarize the top headlines from the provided context. Your output MUST be highly dense, exciting, and exactly between 130 and 150 words maximum so it takes exactly one minute to read aloud."
    else:
        system_instruction = "You are a helpful Voice News Assistant. Use the provided news context to answer the user's query. Keep your answer brief and conversational."
    
    # Create the prompt manually
    prompt_text = f"""{system_instruction}

News Context:
{context}

User Query: {query}"""
    
    messages = [{"role": "user", "content": prompt_text}]
    
    print("Generating summary...")
    try:
        # Using Qwen 2.5 which is the default free model on HuggingChat
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return "Error generating summary."

if __name__ == "__main__":
    from dotenv import load_dotenv
    # Fix for running the script directly vs running from root
    try:
        from rag.retrieve import retrieve_context
    except ModuleNotFoundError:
        from retrieve import retrieve_context

    load_dotenv() # Load variables from .env if it exists
    
    sample_query = "Give me today's AI news?"
    
    # 1. Retrieve context from Chroma DB
    print(f"Retrieving context for query: '{sample_query}'")
    try:
        # Using the optimal database size (100) that we discovered!
        context, docs = retrieve_context(sample_query, persist_directory="chroma_db_100")
        print(f"Retrieved context: {context}\n")
        
        # 2. Summarize the context using the Hugging Face model
        summary = build_prompt_and_summarize(sample_query, context)
        print("\n--- Final Summary ---")
        print(summary)
    except Exception as e:
        import traceback
        import os
        print("\n[Error occurred during summarization!]")
        print("This almost always happens if the Hugging Face token in your .env file is missing, invalid, or still set to the placeholder text.")
        
        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "None")
        print(f"Token currently loaded starts with: '{str(token)[:8]}...'")
        
        print("\nFull Technical Error Details:")
        traceback.print_exc()
