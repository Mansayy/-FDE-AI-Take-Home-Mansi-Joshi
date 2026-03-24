"""Central configuration — reads from .env file at project root."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Locate and load .env from the project root (two levels up from this file)
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR: Path = _project_root
PDF_DIR: Path = BASE_DIR / "pdfs"
DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DIR: Path = DATA_DIR / "chroma"

# ─── Embedding ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "pdf_documents")

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1500"))    # characters
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# ─── LLM ──────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

# ─── Retrieval ────────────────────────────────────────────────────────────────
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "8"))
MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.28"))
