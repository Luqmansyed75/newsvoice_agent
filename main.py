"""
main.py — Voice News Summarizer Agent

Complete Flow:
    1. Load all AI models ONCE at startup (Whisper via stt module singleton;
       HuggingFace Embeddings created here and reused throughout).
    2. Record audio with VAD — stops automatically on silence (no fixed timer).
    3. Convert speech to text (Whisper — already in memory).
    4. Extract search keywords via Hugging Face LLM.
    5. Fetch live news + embed into IN-MEMORY ChromaDB (reuses loaded embeddings).
    6. Retrieve relevant chunks from in-memory vectorstore.
    7. Summarize with Hugging Face LLM.
    8. Speak the answer back (gTTS + pygame).
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ── Import modules ───────────────────────────────────────────────────────────
from config import settings
from voice.stt import speech_to_text, _get_whisper_model   # warm-up import
from voice.tts import text_to_speech
from rag.embed import embed_documents
from rag.retrieve import retrieve_context
from rag.prompt import build_prompt_and_summarize, extract_keywords


def _load_embeddings_model():
    """Load the HuggingFace embedding model once at startup."""
    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"[Loading embedding model '{settings.EMBEDDING_MODEL}' — one-time cost...]")
    emb = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    print("[Embedding model ready!]")
    return emb


def main():
    """
    Main orchestration: Voice Input → RAG Retrieval → LLM Summary → Voice Output.
    All heavy AI models are loaded once at the start, then reused.
    """
    print("=" * 55)
    print("  [Mic]  Voice News Summarizer Agent")
    print("=" * 55)

    # ─── One-time model loading ───────────────────────────────────────────────
    # Whisper singleton is initialised the first time _get_whisper_model() is
    # called.  We trigger it here so the user never waits mid-conversation.
    print("\n[Starting up — loading AI models (done only once)...]")
    _get_whisper_model()          # Whisper STT
    embeddings = _load_embeddings_model()   # HuggingFace sentence-transformer
    print("[All models ready! Let's go.]\n")

    # ─── Mode detection ───────────────────────────────────────────────────────
    is_1min_mode = "--1min" in sys.argv
    mode = "standard"

    if is_1min_mode:
        print("--- [Timer]  1-Minute News Mode (--1min flag) ---")
        query = "Give me a quick digest of all the top headlines."
        mode  = "1min"
    else:
        print("This agent will:")
        print("  1. Listen to your voice question (stops when you go silent)")
        print("  2. Search the live news database")
        print("  3. Summarize the relevant news")
        print("  4. Speak the answer back to you\n")

        # ── Step 1: Speech → Text (VAD recording + cached Whisper) ───────────
        print("--- Step 1: Listening to your voice ---")
        query = speech_to_text()   # stops automatically on silence

        if not query or not query.strip():
            print("Could not understand your speech. Please try again.")
            return

        print(f'\nYour question: "{query}"\n')

        # ── Dynamic mode detection from voice ─────────────────────────────────
        q = query.lower()
        if "story" in q:
            print("--- [Book] Story Mode Activated ---")
            mode = "story"
        elif "one minute" in q or "1 minute" in q:
            print("--- [Timer]  1-Minute Mode Activated ---")
            mode = "1min"

    # ─── Step 1.5: Keyword extraction ─────────────────────────────────────────
    search_query = extract_keywords(query)
    print(f"\n[AI] Extracted search keywords: '{search_query}'")

    # ─── Step 2: Fetch live news + embed into memory (pass shared embeddings) ─
    print(f"\n[Fetching] Hunting the internet for news on: '{search_query}'...")
    try:
        vectorstore = embed_documents(
            query=search_query,
            raw_query=query,                # ← original voice query for date detection
            embeddings=embeddings,          # ← reuse already-loaded model
        )
        print("[Fetch complete — vectorstore is in memory]")
    except Exception as e:
        print(f"\nFailed to fetch live news: {e}")
        print("Check your NEWS_API_KEY in the .env file! Exiting...")
        return

    # ─── Step 3: Retrieve relevant chunks from in-memory vectorstore ──────────
    print("--- Step 3: Searching the in-memory news database ---")
    context, docs = retrieve_context(
        query,
        vectorstore=vectorstore,            # ← pass in-memory store directly
    )

    if not context.strip() or context.strip() == "No news articles found for your specific query.":
        response = "I'm sorry, I couldn't find any recent news on that topic. Please try a different question."
        print(f"\n[No relevant articles found, skipping LLM call]\n")
    else:
        print(f"Found {len(docs)} relevant chunks.\n")

        # ── Step 4: LLM summarization ─────────────────────────────────────────
        print(f"--- Step 4: Generating AI summary (mode: {mode}) ---")
        response = build_prompt_and_summarize(query, context, mode=mode)
        print(f"\nAI Response: {response}\n")

    # ─── Step 5: Text → Speech ────────────────────────────────────────────────
    print("--- Step 5: Speaking the response ---")
    text_to_speech(response)

    print("\n" + "=" * 55)
    print("  Done! Run again to ask another question.")
    print("=" * 55)


if __name__ == "__main__":
    main()
