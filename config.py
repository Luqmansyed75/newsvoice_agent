import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # ── API Keys ─────────────────────────────────────────────────────
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "")
    HF_TOKEN: str = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")

    # ── RAG / Vector DB ──────────────────────────────────────────────
    CHROMA_DIR: str = "chroma_db_300"   # fallback for standalone testing only
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50
    RETRIEVE_K: int = 2

    # ── AI Models ────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "Qwen/Qwen2.5-72B-Instruct"
    WHISPER_MODEL_SIZE: str = "base"

    # ── Voice / Audio ────────────────────────────────────────────────
    SAMPLE_RATE: int = 16000          # Hz — what Whisper expects
    SILENCE_THRESHOLD: float = 0.01   # RMS amplitude below = silence
    SILENCE_DURATION: float = 1.5     # Seconds of silence before stopping
    MAX_RECORD_SECONDS: int = 15      # Hard cap so it never records forever

settings = Settings()
