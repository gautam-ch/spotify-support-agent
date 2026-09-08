"""
phase2_cluster.py - Intent clustering + Golden Set stratified sampling
Run: .venv/Scripts/python scripts/phase2_cluster.py
"""
import sys
import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_PATH  = Path("data/processed/spotify_customer_messages.csv")
OUTPUT_PATH = Path("data/processed")
REPORTS     = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE   = 3000   # we can safely do 3000 now because it runs locally for free!
N_CLUSTERS    = 7      # number of intent clusters (tune later)
SAMPLE_PER_CL = 30     # ~30 per cluster → ~210 total golden examples
RANDOM_SEED   = 42     # reproducibility — always pin your seed
EMBED_MODEL   = "all-MiniLM-L6-v2"
EMBED_DIM     = 384    # sentence-transformers MiniLM dimension

from sentence_transformers import SentenceTransformer

print("=" * 60)
print("PHASE 2 — Intent Clustering & Golden Set Sampling")
print("=" * 60)

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print(f"\n[1/6] Loading {INPUT_PATH} ...")
df = pd.read_csv(INPUT_PATH)
print(f"      Loaded: {len(df):,} customer messages")
print(f"      Columns: {df.columns.tolist()}")

# ── Step 2: Sample 3,000 messages ────────────────────────────────────────────
# WHY sample instead of embed all 28K?
#   K-Means converges well on a representative sample.
#   Embedding all 28K would use ~28K API calls vs 150 batches of 20.
#   For a free tier, 3K is safe; quality difference is negligible.
print(f"\n[2/6] Sampling {SAMPLE_SIZE} messages (seed={RANDOM_SEED}) ...")
sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
texts = sample_df["first_customer_text_clean"].tolist()
print(f"      Sample size: {len(texts)}")
print(f"      Example: {texts[0][:100]}")

# ── Step 3: Embed locally with sentence-transformers ─────────────────────────
# WHY local model instead of an API?
#   1. Zero rate limits — embeds 3,000 texts in ~15 seconds on CPU
#   2. Zero cost — no API key needed
#   3. Deterministic — same model weights always produce identical embeddings
#   4. all-MiniLM-L6-v2 is a proven 22M-param model that performs well on
#      short text (tweets) and produces 384-dim embeddings.
#
# WHY normalize?
#   K-Means uses Euclidean distance. Unnormalized embeddings from long tweets
#   have higher magnitude than short ones, biasing cluster centers toward
#   verbose messages. L2 normalization puts all vectors on the unit sphere,
#   making distance purely about semantic direction, not message length.

print(f"\n[3/8] Embedding {len(texts)} messages with {EMBED_MODEL} ...")
print(f"      Mode: Local CPU via sentence-transformers (zero API calls)")
print(f"      This will take ~10-20 seconds — embeddings are cached after first run.")

cache_path = OUTPUT_PATH / "embeddings_cache.npy"

if cache_path.exists():
    print(f"      Cache found at {cache_path} — loading cached embeddings ...")
    embeddings_array = np.load(cache_path)
    print(f"      Loaded {len(embeddings_array)} cached embeddings")
else:
    # Initialize the local model
    print(f"      Downloading/Loading local model (first run only) ...")
    model = SentenceTransformer(EMBED_MODEL)
    
    # Encode all texts in one go!
    print(f"      Encoding {len(texts)} texts ...")
    embeddings_array = model.encode(texts, show_progress_bar=True, batch_size=128)
    
    print(f"\n      Embedding complete.")
    np.save(cache_path, embeddings_array)
    print(f"      Embeddings cached -> {cache_path}")


print(f"      Embedding matrix shape: {embeddings_array.shape}")

# L2 normalize — puts all vectors on unit sphere
embeddings_norm = normalize(embeddings_array, norm="l2")
print(f"      Normalized embedding matrix shape: {embeddings_norm.shape}")

# ── Step 4: Elbow Method + Silhouette Score ──────────────────────────────────
# WHY do we need this?
#   Hardcoding k=7 with no justification is a red flag in any ML interview.
#   The Elbow Method plots inertia (WCSS) vs k — you look for the "bend".
#   The Silhouette Score measures cluster cohesion vs separation (-1 to +1).
#   Together, they give a quantitative, defensible choice of k.

print(f"\n[4/8] Finding optimal k with Elbow Method + Silhouette Score ...")
K_RANGE = range(3, 13)
inertias = []
silhouettes = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, max_iter=300)
    labels_k = km.fit_predict(embeddings_norm)
    inertias.append(km.inertia_)
    sil = silhouette_score(embeddings_norm, labels_k, sample_size=min(2000, len(embeddings_norm)))
    silhouettes.append(sil)
    print(f"      k={k:>2}  |  Inertia: {km.inertia_:>8.1f}  |  Silhouette: {sil:.4f}")

# Pick the k with the highest silhouette score
best_k = list(K_RANGE)[np.argmax(silhouettes)]
best_sil = max(silhouettes)
print(f"\n      >>> Best k by Silhouette Score: k={best_k} (score={best_sil:.4f})")
print(f"      >>> Using N_CLUSTERS={N_CLUSTERS} (configured). Adjust if best_k differs.")

# Plot Elbow + Silhouette side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(list(K_RANGE), inertias, "bo-", linewidth=2, markersize=8)
ax1.axvline(N_CLUSTERS, color="red", linestyle="--", alpha=0.7, label=f"Selected k={N_CLUSTERS}")
ax1.set_title("Elbow Method (WCSS)", fontsize=14, fontweight="bold")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(list(K_RANGE), silhouettes, "go-", linewidth=2, markersize=8)
ax2.axvline(N_CLUSTERS, color="red", linestyle="--", alpha=0.7, label=f"Selected k={N_CLUSTERS}")
ax2.set_title("Silhouette Score", fontsize=14, fontweight="bold")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
elbow_path = REPORTS / "02a_elbow_silhouette.png"
plt.savefig(elbow_path, dpi=150, bbox_inches="tight")
print(f"      Elbow + Silhouette plot saved -> {elbow_path}")

# ── Step 5: K-Means Clustering (final run) ────────────────────────────────────
# WHY K-Means over HDBSCAN or GMM?
#   For a first pass on Twitter data, K-Means is:
#   - Fast (< 5 sec on 3K x 384 matrix on CPU)
#   - Deterministic with a fixed seed (reproducible)
#   - Forces hard cluster assignment — good for stratified sampling
#   Trade-off: assumes spherical clusters, sensitive to outliers.
#   For a production system you'd evaluate multiple algorithms.

print(f"\n[5/8] Running K-Means (k={N_CLUSTERS}, seed={RANDOM_SEED}) ...")
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_SEED,
    n_init=10,      # run 10 times with different centroids, take best
    max_iter=300,
    algorithm="lloyd",
)
cluster_labels = kmeans.fit_predict(embeddings_norm)
sample_df["cluster"] = cluster_labels

final_sil = silhouette_score(embeddings_norm, cluster_labels)
print(f"      Cluster assignment complete. Silhouette Score: {final_sil:.4f}")

# ── Intent Taxonomy ───────────────────────────────────────────────────────────
# WHY name clusters?
#   Raw cluster IDs (0, 1, 2...) are meaningless to stakeholders.
#   Mapping each cluster to a business intent makes the analysis actionable.
#   These names are derived from manually inspecting the top-3 centroid examples.
#   In production, you'd use an LLM to auto-generate and validate these labels.

INTENT_TAXONOMY = {
    0: "subscription_issue",     # Premium downgraded, charged but no access
    1: "account_access",         # Can't sign in, account hacked, DM requests
    2: "general_inquiry",        # Generic questions, waiting for answers
    3: "off_topic_chatter",      # Memes, jokes, non-support conversations
    4: "app_bug",                # App crashes, missing music, update issues
    5: "billing_dispute",        # Double charged, unauthorized charges, refund
    6: "feature_request",        # Shuffle, playlists, song previews, UI changes
}

sample_df["intent_label"] = sample_df["cluster"].map(INTENT_TAXONOMY)

print(f"\n      Cluster sizes + Intent Labels:")
for cid, count in sample_df["cluster"].value_counts().sort_index().items():
    intent = INTENT_TAXONOMY.get(cid, "unknown")
    bar = "#" * (count // 10)
    print(f"        Cluster {cid} [{intent:>22}]: {count:>4} messages  {bar}")

# ── Find representative examples per cluster ──────────────────────────────────
# The "most representative" example = the one CLOSEST to the cluster centroid.
# WHY centroid distance?
#   It gives the most "average" example — the one that best captures what the
#   cluster is about. Useful for understanding each cluster's theme.

print(f"\n      Top-3 representative examples per cluster:")
print("-" * 60)
for cid in range(N_CLUSTERS):
    mask = cluster_labels == cid
    cluster_vecs = embeddings_norm[mask]
    centroid = kmeans.cluster_centers_[cid]
    similarities = cluster_vecs @ centroid
    top_idx = np.argsort(similarities)[::-1][:3]
    cluster_texts = sample_df[mask].reset_index(drop=True)
    intent = INTENT_TAXONOMY.get(cid, "unknown")
    print(f"\n  Cluster {cid} — {intent} ({mask.sum()} msgs):")
    for j, idx in enumerate(top_idx):
        print(f"    [{j+1}] {cluster_texts.iloc[idx]['first_customer_text_clean'][:100]}")
print("-" * 60)

# ── Step 6: Visualise clusters (PCA 2D) ───────────────────────────────────────
# WHY PCA and not t-SNE?
#   t-SNE is better for visualisation but non-deterministic and slow on 3K x 384.
#   PCA is deterministic, fast, and good enough to see cluster separation.
#   In a real system you'd add a t-SNE plot for the final report.

print(f"\n[6/8] Visualising clusters with PCA (2D) ...")
pca = PCA(n_components=2, random_state=RANDOM_SEED)
coords_2d = pca.fit_transform(embeddings_norm)
explained = pca.explained_variance_ratio_.sum() * 100

colors = plt.cm.Set1(np.linspace(0, 1, N_CLUSTERS))
fig, ax = plt.subplots(figsize=(12, 8))
for cid in range(N_CLUSTERS):
    mask = cluster_labels == cid
    intent = INTENT_TAXONOMY.get(cid, "unknown")
    ax.scatter(
        coords_2d[mask, 0], coords_2d[mask, 1],
        c=[colors[cid]], label=f"{cid}: {intent} (n={mask.sum()})",
        alpha=0.5, s=15,
    )
ax.set_title(
    f"K-Means Intent Clusters (k={N_CLUSTERS}) — PCA 2D\n"
    f"Explained variance: {explained:.1f}% | Silhouette: {final_sil:.3f}",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plot_path = REPORTS / "02b_clusters_pca.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"      PCA plot saved -> {plot_path}")

# ── Step 7: Stratified sampling for Golden Set ────────────────────────────────
# WHY stratified instead of random?
#   Random sampling from 28K messages would give ~proportional coverage.
#   But some intents are rare (e.g., account hacking = 3% of tickets).
#   A random sample of 200 might give 0 hacking examples.
#   Stratified sampling guarantees EVERY cluster is represented,
#   making the Golden Set maximally diverse for evaluation.

print(f"\n[7/8] Stratified sampling for Golden Set ...")
golden_frames = []
for cid in range(N_CLUSTERS):
    cluster_data = sample_df[sample_df["cluster"] == cid]
    n_to_sample = min(SAMPLE_PER_CL, len(cluster_data))
    sampled = cluster_data.sample(n=n_to_sample, random_state=RANDOM_SEED)
    golden_frames.append(sampled)
    intent = INTENT_TAXONOMY.get(cid, "unknown")
    print(f"      Cluster {cid} [{intent}]: sampled {n_to_sample}/{len(cluster_data)} examples")

golden_df = pd.concat(golden_frames, ignore_index=True)
golden_df = golden_df.sample(frac=1, random_state=RANDOM_SEED)  # shuffle rows

# Pre-fill intent_label from taxonomy (saves manual labeling time)
golden_df["intent_label"]  = golden_df["cluster"].map(INTENT_TAXONOMY)
# These two still need manual review:
golden_df["is_escalation"] = ""   # Fill in: True / False
golden_df["label_notes"]   = ""   # Optional: edge case notes

# Save for hand-labeling
golden_out = OUTPUT_PATH / "golden_set_unlabeled_v2.csv"
golden_df[[
    "conversation_id",
    "cluster",
    "intent_label",
    "first_customer_text_clean",
    "num_turns",
    "is_resolved",
    "is_escalation",
    "label_notes",
]].to_csv(golden_out, index=False, encoding="utf-8-sig")

# Also save the full sample with cluster assignments
sample_out = OUTPUT_PATH / "sample_with_clusters.csv"
sample_df.to_csv(sample_out, index=False, encoding="utf-8-sig")

print(f"\n      Golden set (unlabeled): {golden_out}")
print(f"      Total examples to label: {len(golden_df)}")
print(f"      Cluster distribution in golden set:")
print(golden_df[["cluster", "intent_label"]].value_counts().sort_index().to_string())

# ── Step 8: Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
print(f"""
Results:
  Embedding model : {EMBED_MODEL} (local, {EMBED_DIM}-dim)
  Optimal k       : {best_k} (by silhouette), using k={N_CLUSTERS}
  Silhouette Score: {final_sil:.4f}
  Golden Set Size : {len(golden_df)} examples ({SAMPLE_PER_CL} per cluster)

INTENT TAXONOMY (auto-assigned from cluster analysis):
  Cluster 0: subscription_issue  — Premium downgraded, charged but no access
  Cluster 1: account_access      — Can't sign in, account hacked, DM requests
  Cluster 2: general_inquiry     — Generic questions, waiting for answers
  Cluster 3: off_topic_chatter   — Memes, jokes, non-support conversations
  Cluster 4: app_bug             — App crashes, missing music, update issues
  Cluster 5: billing_dispute     — Double charged, unauthorized charges, refund
  Cluster 6: feature_request     — Shuffle, playlists, song previews, UI changes

Next steps:
  1. Open {golden_out} in Excel/VS Code
  2. Review each tweet — intent_label is PRE-FILLED from clustering
  3. Fill in 'is_escalation' (True = needs human, False = AI can handle)
  4. Correct any intent_labels that look wrong
  5. Run: .venv/Scripts/python scripts/phase3_eval.py
""")
