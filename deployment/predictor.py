"""Loads the deployment model chosen by the report stage
(outputs/models/deployment_choice.json) and scores new posts with the SAME
pre-processing used during the experiments."""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
import config  # noqa: E402
from src.preprocessing import clean_text, process_for_classical  # noqa: E402


class CyberbullyingDetector:
    def __init__(self, choice_path=None, which="recommended"):
        choice_path = Path(choice_path or config.OUTPUT_DIR / "models" / "deployment_choice.json")
        if not choice_path.exists():
            raise FileNotFoundError(f"{choice_path} not found - run `python run_all.py --stages report` first.")
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        spec = choice[which] or choice["recommended"]
        self.thresholds = choice["thresholds"]
        self.model_name, self.family = spec["model"], spec["family"]
        self.train_dataset = spec["train_dataset"]
        artifact = Path(spec["artifact"])
        if self.family == "classical":
            import joblib
            b = joblib.load(artifact)
            self.vectorizer, self.model, self.labels = b["vectorizer"], b["model"], b["labels"]
        elif self.family == "deep_learning":
            from src.bilstm_model import _select_backend, texts_to_sequences
            keras = _select_backend()
            meta = json.loads((artifact / "vocab.json").read_text(encoding="utf-8"))
            self.model = keras.saving.load_model(artifact / "model.keras")
            self.vocab, self.max_len, self.labels = meta["vocab"], meta["max_len"], meta["labels"]
            self._to_seq = texts_to_sequences
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(artifact)
            self.model = AutoModelForSequenceClassification.from_pretrained(artifact).to(self.device).eval()
            self.labels = config.BINARY_LABELS

    def predict_proba(self, texts):
        cleaned = [clean_text(t) for t in texts]
        if self.family == "classical":
            return self.model.predict_proba(self.vectorizer.transform([process_for_classical(t) for t in cleaned]))
        if self.family == "deep_learning":
            X = self._to_seq([process_for_classical(t) for t in cleaned], self.vocab, self.max_len)
            return self.model.predict(X, verbose=0)
        with self.torch.no_grad():
            enc = self.tokenizer(cleaned, padding=True, truncation=True,
                                 max_length=config.TRANSFORMER["max_len"], return_tensors="pt").to(self.device)
            return self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()

    def action(self, p):
        if p >= self.thresholds["auto_hide"]:
            return "auto_hide"
        if p >= self.thresholds["flag_for_review"]:
            return "flag_for_human_review"
        return "allow"

    def predict(self, texts):
        proba = np.asarray(self.predict_proba(texts))
        pos = self.labels.index("cyberbullying")
        if self.family == "classical":  # SVM decisions come from its decision function
            X = self.vectorizer.transform([process_for_classical(clean_text(t)) for t in texts])
            pred = self.model.predict(X)
        else:
            pred = proba.argmax(axis=1)
        return [{"text": t, "cyberbullying_probability": round(float(p[pos]), 4),
                 "label": self.labels[int(k)], "action": self.action(float(p[pos]))}
                for t, p, k in zip(texts, proba, pred)]
