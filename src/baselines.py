"""
src/baselines.py - Trivial and Simple Baselines for Support Agent Evaluation

Implements two standard comparison baselines:
1. Baseline 1 (Trivial):
   - Intent: Majority class prediction.
   - Escalation: Never escalate (all auto-handled).
   - Reply: Static generic template response.

2. Baseline 2 (Simple):
   - Intent: TF-IDF vectorizer + Cosine Similarity Nearest Neighbor.
   - Escalation: Keyword-based heuristic rules.
   - Reply: Verbatim retrieval of the nearest neighbor's historical Spotify reply.
"""
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_CLUSTERS_CSV = DEFAULT_PROCESSED_DIR / "sample_with_clusters.csv"
DEFAULT_THREADS_CSV = DEFAULT_PROCESSED_DIR / "spotify_threads.csv"

# Escalation trigger keywords for Simple Baseline
ESCALATION_KEYWORDS = [
    "hack", "hacked", "stolen", "compromised",
    "unauthorized", "charge", "charged", "refund",
    "double bill", "billed twice", "fraud", "banned",
    "password reset", "cant login", "can't login", "cannot log in",
    "cant sign in", "can't sign in", "cannot sign in",
]


class TrivialBaselineAgent:
    """
    Baseline 1: Trivial Baseline.
    Always predicts majority intent, never escalates, returns static template.
    """

    def __init__(self, majority_intent: str = "general_inquiry"):
        self.majority_intent = majority_intent
        self.template_reply = (
            "Hi there! Thanks for reaching out to Spotify Cares. We're here to help! "
            "Could you please send us a direct message with your account email and more details "
            "so our team can look into this for you?"
        )

    def process(self, customer_tweet: str) -> Dict[str, Any]:
        return {
            "system": "Baseline 1 (Trivial)",
            "customer_tweet": customer_tweet,
            "predicted_intent": self.majority_intent,
            "confidence": 1.0,
            "is_escalation": False,
            "escalation_reason": "Trivial rule: never escalate any tickets.",
            "draft_reply": self.template_reply,
            "retrieved_context": "",
        }


class SimpleBaselineAgent:
    """
    Baseline 2: Simple Baseline.
    Uses TF-IDF Nearest Neighbor for intent & reply retrieval, plus keyword matching for escalation.
    """

    def __init__(
        self,
        clusters_csv: Path = DEFAULT_CLUSTERS_CSV,
        threads_csv: Path = DEFAULT_THREADS_CSV,
    ):
        if not clusters_csv.exists():
            raise FileNotFoundError(f"Clusters dataset not found at {clusters_csv}")

        sample_df = pd.read_csv(clusters_csv)
        if threads_csv.exists():
            threads_df = pd.read_csv(
                threads_csv,
                usecols=["conversation_id", "last_spotify_reply_clean"],
            )
            merged = sample_df.merge(threads_df, on="conversation_id", how="left")
            merged["last_spotify_reply_clean"] = merged[
                "last_spotify_reply_clean"
            ].fillna("Hi there! Please send us a DM with your account details.")
        else:
            merged = sample_df
            merged["last_spotify_reply_clean"] = "Hi there! Please send us a DM with your account details."

        self.corpus_df = merged
        self.texts = self.corpus_df["first_customer_text_clean"].fillna("").tolist()

        # Fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def process(self, customer_tweet: str) -> Dict[str, Any]:
        tweet_lower = customer_tweet.lower()

        # 1. Escalation via keyword heuristics
        matched_keywords = [kw for kw in ESCALATION_KEYWORDS if kw in tweet_lower]
        if matched_keywords:
            is_escalation = True
            escalation_reason = (
                f"Keyword heuristic triggered by terms: {', '.join(matched_keywords)}"
            )
        else:
            is_escalation = False
            escalation_reason = "No escalation keywords detected; auto-handled."

        # 2. Intent & Reply via TF-IDF Nearest Neighbor
        query_vec = self.vectorizer.transform([customer_tweet])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])

        matched_row = self.corpus_df.iloc[best_idx]
        predicted_intent = str(matched_row.get("intent_label", "general_inquiry"))
        retrieved_reply = str(matched_row.get("last_spotify_reply_clean", ""))

        return {
            "system": "Baseline 2 (Simple TF-IDF)",
            "customer_tweet": customer_tweet,
            "predicted_intent": predicted_intent,
            "confidence": round(best_sim, 4),
            "is_escalation": is_escalation,
            "escalation_reason": escalation_reason,
            "draft_reply": retrieved_reply,
            "retrieved_context": f"Matched nearest neighbor (sim={best_sim:.2f}): {matched_row.get('first_customer_text_clean', '')}",
        }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Initializing Baselines...")
    trivial = TrivialBaselineAgent()
    simple = SimpleBaselineAgent()

    test_tweet = "@SpotifyCares I got charged twice for my family plan! Please refund."

    print("\n--- Test Tweet ---")
    print(test_tweet)

    print("\n--- Trivial Baseline Output ---")
    print(trivial.process(test_tweet))

    print("\n--- Simple Baseline Output ---")
    print(simple.process(test_tweet))
