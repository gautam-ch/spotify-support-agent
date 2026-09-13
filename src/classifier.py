"""
src/classifier.py - Intent Classifier

Classifies incoming customer support messages into the 7 defined intent categories
discovered in Phase 2 using embedding features and calibrated classification.
"""
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

def get_default_processed_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "processed",
        Path("data/processed"),
        Path(__file__).resolve().parent.parent / "processed",
        Path("processed"),
        Path(__file__).resolve().parent.parent,
        Path("."),
    ]
    for c in candidates:
        if (c / "sample_with_clusters.csv").exists():
            return c
    return Path(__file__).resolve().parent.parent / "data" / "processed"

DEFAULT_PROCESSED_DIR = get_default_processed_dir()
DEFAULT_CLUSTERS_CSV = DEFAULT_PROCESSED_DIR / "sample_with_clusters.csv"
DEFAULT_EMBEDDINGS_NPY = DEFAULT_PROCESSED_DIR / "embeddings_cache.npy"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

INTENT_LABELS = [
    "subscription_issue",
    "account_access",
    "general_inquiry",
    "off_topic_chatter",
    "app_bug",
    "billing_dispute",
    "feature_request",
]


class IntentClassifier:
    """
    Supervised Intent Classifier trained on the 3,000 clustered representations
    from Phase 2 using MiniLM embeddings.
    """

    def __init__(
        self,
        clusters_csv: Path = DEFAULT_CLUSTERS_CSV,
        embeddings_npy: Path = DEFAULT_EMBEDDINGS_NPY,
        model_name: str = DEFAULT_MODEL_NAME,
        retriever_model: SentenceTransformer = None,
    ):
        self.model_name = model_name
        self.encoder = retriever_model if retriever_model is not None else SentenceTransformer(model_name)

        if not clusters_csv.exists():
            raise FileNotFoundError(f"Clusters dataset not found at {clusters_csv}")
        df = pd.read_csv(clusters_csv)

        if not embeddings_npy.exists():
            raise FileNotFoundError(f"Embeddings cache not found at {embeddings_npy}")
        embeddings = np.load(embeddings_npy)

        X = normalize(embeddings)
        y = df["intent_label"].values

        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.classifier.fit(X, y)
        self.classes_ = list(self.classifier.classes_)

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict intent for a single customer message.
        """
        emb = self.encoder.encode([text])
        emb_norm = normalize(emb)
        probs = self.classifier.predict_proba(emb_norm)[0]

        top_idx = int(np.argmax(probs))
        predicted_intent = self.classes_[top_idx]
        confidence = float(probs[top_idx])

        prob_dict = {cls: float(round(p, 4)) for cls, p in zip(self.classes_, probs)}

        return {
            "intent": predicted_intent,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    def predict_batch(self, texts: List[str], batch_size: int = 64) -> List[Dict[str, Any]]:
        """
        Predict intents for a batch of customer messages.
        """
        if not texts:
            return []
        embs = self.encoder.encode(texts, batch_size=batch_size, show_progress_bar=False)
        embs_norm = normalize(embs)
        all_probs = self.classifier.predict_proba(embs_norm)

        results = []
        for probs in all_probs:
            top_idx = int(np.argmax(probs))
            results.append({
                "intent": self.classes_[top_idx],
                "confidence": float(round(probs[top_idx], 4)),
                "probabilities": {cls: float(round(p, 4)) for cls, p in zip(self.classes_, probs)},
            })
        return results


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Initializing IntentClassifier...")
    clf = IntentClassifier()

    test_queries = [
        "@SpotifyCares I can't log in to my account since yesterday!",
        "Can you please add a bigger shuffle button to the desktop player?",
        "I was charged twice on my credit card for Spotify Family!",
        "The app keeps crashing whenever I try to play offline music on iOS.",
        "Hey @SpotifyCares what are you guys up to today haha",
    ]

    print("\nRunning test predictions:")
    print("-" * 60)
    for q in test_queries:
        res = clf.predict(q)
        print(f"Query: {q}")
        print(f"  -> Predicted Intent: {res['intent']} (Confidence: {res['confidence']:.2f})")
        print()
