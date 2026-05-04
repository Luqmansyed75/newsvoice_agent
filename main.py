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
from dotenv import load_dotenv

# Load environment variables (.env file contains our Hugging Face token)
load_dotenv()

# Import our custom modules
from voice.stt import speech_to_text
from voice.tts import text_to_speech
from rag.retrieve import retrieve_context
from rag.prompt import build_prompt_and_summarize


def main():
    """
    Main orchestration function that ties together the entire pipeline:
    Voice Input -> RAG Retrieval -> LLM Summary -> Voice Output
    """
    print("=" * 50)
    print("  Voice News Summarizer Agent")
    print("=" * 50)
    print("\nThis agent will:")
    print("  1. Listen to your voice question")
    print("  2. Search the news database")
    print("  3. Summarize the relevant news")
    print("  4. Speak the answer back to you\n")
    
    # ─── Step 1: Speech-to-Text (Record + Transcribe) ───
    print("--- Step 1: Listening to your voice ---")
    query = speech_to_text(duration=5, model_size="base")
    
    # Check if Whisper captured anything meaningful
    if not query or query.strip() == "":
        print("Could not understand your speech. Please try again.")
        return
    
    print(f'\nYour question: "{query}"\n')
    
    # ─── Step 2: Retrieve relevant news from ChromaDB ───
    print("--- Step 2: Searching the news database ---")
    context, docs = retrieve_context(query, persist_directory="chroma_db_100")
    print(f"Found {len(docs)} relevant chunks.\n")
    
    # ─── Step 3: Generate summary using Hugging Face LLM ───
    print("--- Step 3: Generating AI summary ---")
    response = build_prompt_and_summarize(query, context)
    print(f"\nAI Response: {response}\n")
    
    # ─── Step 4: Text-to-Speech (Speak the response) ───
    print("--- Step 4: Speaking the response ---")
    text_to_speech(response)
    
    print("\n" + "=" * 50)
    print("  Done! Ask another question by running again.")
    print("=" * 50)


if __name__ == "__main__":
    main()
