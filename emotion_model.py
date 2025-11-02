
from typing import List
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pickle

class EmotionModel:
    def __init__(self, model_dir: str):
        model_dir = str(Path(model_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()
        
        # Load the actual emotion labels from the pickle file
        labels_file = Path(model_dir) / "labels.pkl"
        if labels_file.exists():
            with open(labels_file, 'rb') as f:
                self.emotion_labels = pickle.load(f)
            # Create proper id2label mapping
            self.id2label = {i: label for i, label in enumerate(self.emotion_labels)}
        else:
            # Fallback to config labels
            self.id2label = self.model.config.id2label if hasattr(self.model.config, "id2label") else {i: str(i) for i in range(self.model.config.num_labels)}

    @torch.inference_mode()
    def predict(self, text: str, top_k: int = 3, threshold: float = 0.2) -> List[str]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(0)  # multilabel is common for GoEmotions finetunes
        # If your model is multi-label, apply sigmoid; if single-label, apply softmax
        if self.model.config.problem_type == "multi_label_classification" or logits.ndim == 1:
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = []
            for idx in np.argsort(-probs)[:top_k]:
                if probs[idx] >= threshold:
                    labels.append(self.id2label.get(idx, str(idx)))
            return labels
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            idx = int(np.argmax(probs))
            return [self.id2label.get(idx, str(idx))]
