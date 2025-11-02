# MindLens Dockerfile
# - Copies your app and (optionally) your local models into the image
# - Runs the Streamlit UI on port 8501
#
# EXPECTED LOCAL FOLDERS (put them next to this Dockerfile before building):
#   BERT_FINE_TURNED_EMOTION_DECTION_USING_TEXT/  <- your GoEmotions fine-tuned model
#   spacy_model_context/                          <- your spaCy textcat_multilabel directory
#   bert_model_offilne/                           <- your SentenceTransformers model
#   models/small/                                 <- your Whisper local model (optional)
#   sucide_detection/                             <- optional Keras .h5 model & tokenizer.pickle (fallback path kept)
#
# If you don't have some models, leave the folders empty or omit them; the app will
# still run using the rule-based suicide detector fallback.

FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better Docker layer caching
COPY requirements.txt ./

# Install Python deps
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Create data dir (mounted as volume at runtime)
RUN mkdir -p /app/data

# ---- Model paths inside the image ----
# We point config via ENV so paths work inside container, regardless of your local machine.
ENV GOEMOTIONS_MODEL_DIR=/app/BERT_FINE_TURNED_EMOTION_DECTION_USING_TEXT \
    SPACY_MODEL_DIR=/app/spacy_model_context \
    SBERT_MODEL_DIR=/app/bert_model_offilne \
    WHISPER_MODEL_DIR=/app/models/small \
    SUICIDE_MODEL_PATH=/app/sucide_detection/suicide_detection_model.h5 \
    SUICIDE_TOKENIZER_PATH=/app/sucide_detection/tokenizer.pickle \
    # Storage / runtime options
    VECTOR_STORE=faiss \
    LLM_BACKEND=none \
    AUTO_ALERT_ENABLED=false

# Expose Streamlit port
EXPOSE 8501

# Streamlit runs best with a process supervisor (tini) for clean shutdown
ENTRYPOINT ["/usr/bin/tini","-g","--"]

# Default command: run Streamlit UI
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]