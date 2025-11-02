# MindLens – Dockerized

This lets any client run the **MindLens** app with a single Docker image.

## 1) Folder layout (before you build)

Place these (if you have them) next to the Dockerfile:

```
BERT_FINE_TURNED_EMOTION_DECTION_USING_TEXT/  # from your GoEmotions fine-tune
spacy_model_context/                          # your spaCy textcat_multilabel pipeline dir
bert_model_offilne/                           # SentenceTransformers model dir
models/
  small/                                      # Faster-Whisper local model (optional)
sucide_detection/
  suicide_detection_model.h5                  # optional
  tokenizer.pickle                            # optional
```

If you don't have some models, you can still build and run; the app will fall back to the rule-based suicide detector.

> The default paths in code are defined in `config.py` and can be overridden via ENV. For example, `GOEMOTIONS_MODEL_DIR`, `SPACY_MODEL_DIR`, `SBERT_MODEL_DIR`, `WHISPER_MODEL_DIR` (see Dockerfile).

## 2) Build the image

```bash
docker build -t mindlens:latest .
```

## 3) Run it (simple)

```bash
docker run --rm -p 8501:8501 -v mindlens_data:/app/data mindlens:latest
```

Then open: http://localhost:8501

## 4) Or use docker-compose (recommended)

```bash
docker compose up --build
```

This persists everything in the named volume `mindlens_data`.

## 5) Environment variables you might override

- `GOEMOTIONS_MODEL_DIR` (default `/app/BERT_FINE_TURNED_EMOTION_DECTION_USING_TEXT`) – used by `EmotionModel` to load HuggingFace model locally (see `emotion_model.py`).  
- `SPACY_MODEL_DIR` (default `/app/spacy_model_context`) – spaCy pipeline dir used by `ContextTagger` (see `tagger.py`).  
- `SBERT_MODEL_DIR` (default `/app/bert_model_offilne`) – sentence-transformers model for embeddings (see `embedder.py`).  
- `WHISPER_MODEL_DIR` (default `/app/models/small`) – Faster-Whisper local model for audio/video transcription (used in `app.py`).  
- `SUICIDE_MODEL_PATH` and `SUICIDE_TOKENIZER_PATH` – optional Keras model; **the app currently uses a rule-based fallback** (see `suicide_detector_fallback.py`).  
- `VECTOR_STORE` – `faiss` (default) or `chroma` (see `storage.py`).  
- `LLM_BACKEND` – `none` (default) or `huggingface` to use HF Inference API for summaries (`rag.py`).  
- `HF_API_TOKEN` / `HF_MODEL` – only needed if `LLM_BACKEND=huggingface` (see `config.py`).  
- `AUTO_ALERT_ENABLED` – `false` by default in Docker; WhatsApp Web automation typically **does not work in a headless container**. You can re-enable at your own risk.

## 6) Data & persistence

All user entries, embeddings, FAISS index/metadata are written to `/app/data` (see `config.py` and `storage.py`). The compose file mounts this path to a named volume so nothing is lost when the container stops.

## 7) Notes & caveats

- **WhatsApp alerts** (`pywhatkit`) require a logged-in **desktop browser session**; this is not available inside a headless container, so it is **disabled by default** via `AUTO_ALERT_ENABLED=false`. The rest of the mental-health features (like nearby hospital lookup using OpenStreetMap) work normally.
- If you switch to `VECTOR_STORE=chroma`, make sure the `chromadb` wheel matches your platform and that the container has write access to `/app/data/db`.
- If you want to include all models **inside** the image for offline clients, make sure those model folders exist locally before you build — the Dockerfile will copy them in with `COPY . .`.
- To reduce image size:
  - Remove unused packages from `requirements.txt`.
  - Keep `LLM_BACKEND=none` (this avoids extra runtime downloads).
  - Ensure your model folders only include what is needed (config, tokenizer, weights).

## 8) Troubleshooting

- If Streamlit shows an import error like `MindLensApp` missing, check that `main.py` exists and is copied (it should be).  
- If spaCy fails to load, verify your pipeline directory path and that it contains `meta.json` and `model` artifacts.  
- If FAISS complains about the index, delete `/app/data/faiss_index.faiss` and `/app/data/faiss_meta.jsonl` to rebuild.

Happy journaling!