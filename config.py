import os
from pathlib import Path

# ---------- Paths to your LOCAL models (already downloaded) ----------
# Update these to the actual folders on your laptop.
GOEMOTIONS_MODEL_DIR = os.getenv("GOEMOTIONS_MODEL_DIR", "BERT_FINE_TURNED_EMOTION_DECTION_USING_TEXT")
SPACY_MODEL_DIR      = os.getenv("SPACY_MODEL_DIR",      "spacy_model_context")
SBERT_MODEL_DIR      = os.getenv("SBERT_MODEL_DIR",      "bert_model_offilne")
WHISPER_MODEL_DIR    = os.getenv("WHISPER_MODEL_DIR",    "models/small")
SUICIDE_MODEL_PATH   = os.getenv("SUICIDE_MODEL_PATH",   "sucide_detection/suicide_detection_model.h5")
SUICIDE_TOKENIZER_PATH = os.getenv("SUICIDE_TOKENIZER_PATH", "sucide_detection/tokenizer.pickle")

# ---------- Storage backend ----------
# Options: 'chroma' or 'faiss'
VECTOR_STORE = os.getenv("VECTOR_STORE", "faiss")

# Data dirs
DATA_DIR = Path(os.getenv("DATA_DIR", str(Path.cwd() / "data")))
CHROMA_DIR = DATA_DIR / "db"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.faiss"
FAISS_META_JSONL = DATA_DIR / "faiss_meta.jsonl"
ENTRIES_JSONL     = DATA_DIR / "entries.jsonl"
IMAGES_DIR        = Path(os.getenv("IMAGES_DIR", str(Path.cwd() / "data" / "images")))
VIDEOS_DIR        = Path(os.getenv("VIDEOS_DIR", str(Path.cwd() / "data" / "videos")))

# ---------- Optional LLM summarization ----------
# Choose: 'huggingface' or 'none'
# - 'huggingface': Use Hugging Face Inference API (cloud-based)
# - 'none': Use simple formatted summaries without LLM
LLM_BACKEND = os.getenv("LLM_BACKEND", "none")

# Hugging Face API settings
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# App settings
APP_TITLE = "MindLens — AI‑Powered Digital Diary"
ABOUT_TEXT = os.getenv("ABOUT_TEXT", """
MindLens is your private, AI‑powered digital diary. It detects emotions, tags context,
stores embeddings for powerful search, and lets you query your life with RAG.
Built by <your name>. Replace this text from the About page.
"""
)

# Emergency contacts for suicide risk alerts
EMERGENCY_CONTACTS_JSON = os.getenv("EMERGENCY_CONTACTS_JSON", str(DATA_DIR / "emergency_contacts.json"))
AUTO_ALERT_ENABLED = os.getenv("AUTO_ALERT_ENABLED", "true").lower() == "true"