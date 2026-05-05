"""
main.py — Voice News Summarizer Agent

Complete Flow:
    1. Record audio from mic (sounddevice)
    2. Convert speech to text query (Whisper)
    3. Retrieve relevant news chunks from ChromaDB
    4. Send query + context to Hugging Face LLM for summarization
    5. Convert the LLM response to speech (gTTS)
    6. Play the audio output (pygame)
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables (.env file contains our Hugging Face token)
load_dotenv()

# Import our custom modules
from voice.stt import speech_to_text
from voice.tts import text_to_speech
from rag.embed import embed_documents
from rag.retrieve import retrieve_context
from rag.prompt import build_prompt_and_summarize, extract_keywords

def main():
    """
    Main orchestration function that ties together the entire pipeline:
    Voice Input -> RAG Retrieval -> LLM Summary -> Voice Output
    """
    print("=" * 50)
    print("  Voice News Summarizer Agent")
    print("=" * 50)
    
    # Check if user ran with the --1min flag
    is_1min_mode = "--1min" in sys.argv
    
    # Wait to fetch news until we know what the user wants!
    
    # ─── New Feature: Modes ───
    mode = "standard"
    
    if is_1min_mode:
        print("\n--- ⏱️ 1-Minute News Mode Activated ---")
        query = "Give me a quick digest of all the top headlines."
        mode = "1min"
    else:
        print("\nThis agent will:")
        print("  1. Listen to your voice question")
        print("  2. Search the live news database")
        print("  3. Summarize the relevant news")
        print("  4. Speak the answer back to you\n")
        
        # ─── Step 1: Speech-to-Text (Record + Transcribe) ───
        print("--- Step 1: Listening to your voice ---")
        query = speech_to_text(duration=10, model_size="base")
        
        # Check if Whisper captured anything meaningful
        if not query or query.strip() == "":
            print("Could not understand your speech. Please try again.")
            return
            
        print(f'\nYour question: "{query}"\n')
        
        # Dynamic Mode Detection based on Voice
        query_lower = query.lower()
        if "story" in query_lower:
            print("\n--- 📖 Story Mode Activated by Voice ---")
            mode = "story"
        elif "one minute" in query_lower or "1 minute" in query_lower:
            print("\n--- ⏱️ 1-Minute News Mode Activated by Voice ---")
            mode = "1min"
            
    # ─── Step 1.2: Extract Keywords for Search ───
    search_query = extract_keywords(query)
    print(f"\n[AI] Extracted search keywords: '{search_query}'")
            
    # ─── Step 1.5: Fetch Live News & Update Vector Database based on Query ───
    print(f"\n[Fetching] Hunting the internet for news related to: '{search_query}'...")
    try:
        embed_documents(query=search_query, chunk_size=300, persist_directory="chroma_db_300")
        print("[Fetch Complete]")
    except Exception as e:
        print(f"\nFailed to fetch live news: {e}")
        print("Please check your NEWS_API_KEY in the .env file! Exiting...")
        return
    
    # ─── Step 2: Retrieve relevant news from ChromaDB ───
    print("--- Step 2: Searching the live news database ---")
    context, docs = retrieve_context(query, persist_directory="chroma_db_300")
    print(f"Found {len(docs)} relevant chunks.\n")
    
    # ─── Step 3: Generate summary using Hugging Face LLM ───
    print(f"--- Step 3: Generating AI summary (Mode: {mode}) ---")
    response = build_prompt_and_summarize(query, context, mode=mode)
    print(f"\nAI Response: {response}\n")
    
    # ─── Step 4: Text-to-Speech (Speak the response) ───
    print("--- Step 4: Speaking the response ---")
    text_to_speech(response)
    
    print("\n" + "=" * 50)
    print("  Done! Ask another question by running again.")
    print("=" * 50)


if __name__ == "__main__":
    main()
