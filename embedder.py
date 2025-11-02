
from typing import List
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

class SBERTEmbedder:
    def __init__(self, model_dir: str):
        model_dir = str(Path(model_dir))
        self.model = SentenceTransformer(model_dir)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
