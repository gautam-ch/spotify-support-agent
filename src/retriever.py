"""
src/retriever.py - RAG Historical Resolution Retriever

Retrieves Spotify's historical support resolutions for incoming customer queries
using semantic vector search over the processed customer dataset.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
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

# Default Paths
DEFAULT_PROCESSED_DIR = get_default_processed_dir()
DEFAULT_CLUSTERS_CSV = DEFAULT_PROCESSED_DIR / "sample_with_clusters.csv"
DEFAULT_THREADS_CSV = DEFAULT_PROCESSED_DIR / "spotify_threads.csv"
DEFAULT_EMBEDDINGS_NPY = DEFAULT_PROCESSED_DIR / "embeddings_cache.npy"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SpotifyRAGRetriever:
    """
    RAG Retriever that indexes historical Spotify customer queries and responses.
    Allows fast cosine similarity search over cached MiniLM embeddings.
    """

    def __init__(
        self,
        clusters_csv: Path = DEFAULT_CLUSTERS_CSV,
        threads_csv: Path = DEFAULT_THREADS_CSV,
        embeddings_npy: Path = DEFAULT_EMBEDDINGS_NPY,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # 1. Load sample with clusters
        if not clusters_csv.exists():
            raise FileNotFoundError(f"Clusters dataset not found at {clusters_csv}")
        sample_df = pd.read_csv(clusters_csv)

        # 2. Merge historical Spotify replies from threads
        if threads_csv.exists():
            threads_df = pd.read_csv(
                threads_csv,
                usecols=["conversation_id", "last_spotify_reply_clean"],
            )
            merged_df = sample_df.merge(threads_df, on="conversation_id", how="left")
            merged_df["last_spotify_reply_clean"] = merged_df[
                "last_spotify_reply_clean"
            ].fillna("Hi there! Could you please DM us with your account details?")
        else:
            merged_df = sample_df
            merged_df["last_spotify_reply_clean"] = "No historical reply available."

        self.corpus_df = merged_df

        # 3. Load or compute embeddings
        if embeddings_npy.exists():
            embeddings = np.load(embeddings_npy)
            if len(embeddings) != len(self.corpus_df):
                raise ValueError(
                    f"Embeddings length ({len(embeddings)}) does not match corpus length ({len(self.corpus_df)})"
                )
            self.embeddings_norm = normalize(embeddings)
        else:
            texts = self.corpus_df["first_customer_text_clean"].fillna("").tolist()
            embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True)
            np.save(embeddings_npy, embeddings)
            self.embeddings_norm = normalize(embeddings)

    @staticmethod
    def sanitize_historical_reply(text: str) -> str:
        """Replace dead or deprecated historical shortlinks with modern canonical endpoints."""
        import re
        # Pattern for spoti.fi, bit.ly, t.co shortlinks
        sanitized = re.sub(r"https?://(?:spoti\.fi|bit\.ly|t\.co)/\S+", "https://support.spotify.com", text)
        return sanitized

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Given a new customer message, retrieve top_k most semantically similar
        historical customer issues along with Spotify's actual resolution.
        """
        query_vec = self.model.encode([query])
        query_vec_norm = normalize(query_vec)

        # Dot product with normalized vectors = cosine similarity
        similarities = (query_vec_norm @ self.embeddings_norm.T)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.corpus_df.iloc[idx]
            raw_reply = str(row.get("last_spotify_reply_clean", ""))
            clean_reply = self.sanitize_historical_reply(raw_reply)
            results.append(
                {
                    "rank": rank,
                    "similarity": float(similarities[idx]),
                    "conversation_id": int(row["conversation_id"]),
                    "customer_text": str(row.get("first_customer_text_clean", "")),
                    "spotify_reply": clean_reply,
                    "intent_label": str(row.get("intent_label", "unknown")),
                    "cluster": int(row.get("cluster", -1)),
                    "is_resolved": bool(row.get("is_resolved", False)),
                }
            )
        return results

    def format_context_for_prompt(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved historical threads into a compact, high-signal prompt context.
        """
        if not retrieved_docs:
            return "No historical resolutions found."

        sections = []
        for doc in retrieved_docs:
            sections.append(
                f"[Historical Example {doc['rank']} | Intent: {doc['intent_label']} | Sim: {doc['similarity']:.2f}]\n"
                f"Customer Query: {doc['customer_text']}\n"
                f"Spotify Verified Resolution: {doc['spotify_reply']}"
            )
        return "\n\n".join(sections)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Initializing SpotifyRAGRetriever...")
    retriever = SpotifyRAGRetriever()
    test_query = "@SpotifyCares I was charged twice for Premium this month. Can I get a refund?"
    print(f"\nTest Query: {test_query}\n")

    hits = retriever.retrieve(test_query, top_k=3)
    formatted = retriever.format_context_for_prompt(hits)
    print("Formatted Prompt Context:\n")
    print(formatted)
