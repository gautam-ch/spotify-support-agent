# phase1_run.py — Executes all Phase 1 logic from the notebook as a plain script.
# Run from project root: .venv/Scripts/python scripts/phase1_run.py

import pandas as pd
import numpy as np
import re
import sys
import warnings
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Force UTF-8 output — Windows terminals default to cp1252 which can't encode
# emoji, curly quotes, or many characters that appear in real tweet text.
# Without this, any print() containing a non-ASCII character crashes the script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

DATA_PATH   = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/processed")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
REPORTS     = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

# ── CELL 2 ── Load ────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — Data Engineering & Thread Reconstruction")
print("=" * 60)
print(f"\n[1/7] Loading {DATA_PATH} ...")

df = pd.read_csv(
    DATA_PATH,
    dtype={
        "tweet_id"               : str,
        "in_response_to_tweet_id": str,
        "response_tweet_id"      : str,
    },
    low_memory=False,
)

print(f"      Shape  : {df.shape}")
print(f"      Columns: {df.columns.tolist()}")
print(f"      Memory : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# ── CELL 4 ── Find Spotify author_id ──────────────────────────────────────────
print("\n[2/7] Identifying Spotify's author_id ...")

spotify_text_mask = df["text"].str.contains("SpotifyCares", case=False, na=False)
spotify_outbound  = df[spotify_text_mask & (df["inbound"] == False)]

print(f"      Outbound tweets mentioning SpotifyCares: {len(spotify_outbound):,}")
print("      Top author_ids:")
print(spotify_outbound["author_id"].value_counts().head(5).to_string())

SPOTIFY_AUTHOR_ID = spotify_outbound["author_id"].value_counts().idxmax()
print(f"\n      >>> Spotify's author_id: {SPOTIFY_AUTHOR_ID}")

# ── CELL 5 ── Build Spotify pool ──────────────────────────────────────────────
print("\n[3/7] Building Spotify conversation pool ...")

spotify_replies = df[df["author_id"] == SPOTIFY_AUTHOR_ID].copy()
parent_ids      = spotify_replies["in_response_to_tweet_id"].dropna().unique()
customer_tweets = df[df["tweet_id"].isin(parent_ids)].copy()
spotify_df      = pd.concat([spotify_replies, customer_tweets], ignore_index=True)

print(f"      Tweets sent BY Spotify        : {len(spotify_replies):,}")
print(f"      Unique customer tweets replied : {len(parent_ids):,}")
print(f"      Customer tweets retrieved      : {len(customer_tweets):,}")
print(f"      Total rows in Spotify pool     : {len(spotify_df):,}  ({len(spotify_df)/len(df)*100:.1f}% of full dataset)")

# ── CELL 6a ── Lookup structures ──────────────────────────────────────────────
print("\n[4/7] Building lookup structures ...")

# WHY drop_duplicates here?
# pd.concat([spotify_replies, customer_tweets]) can produce duplicate tweet_ids.
# Example: a tweet sent BY Spotify that was itself a reply — it appears in
# spotify_replies AND gets pulled in as a parent_id of another tweet.
# to_dict(orient='index') requires a unique index, so we deduplicate first.
# We keep the FIRST occurrence (arbitrary but deterministic).
spotify_df = spotify_df.drop_duplicates(subset="tweet_id", keep="first")
print(f"      After dedup: {len(spotify_df):,} unique tweets")

tweet_lookup = spotify_df.set_index("tweet_id").to_dict(orient="index")
print(f"      Lookup table: {len(tweet_lookup):,} entries")

response_map = {}
for tid, row in tweet_lookup.items():
    resp = row.get("response_tweet_id")
    if pd.notna(resp) and str(resp) != "nan":
        response_map[tid] = [r.strip() for r in str(resp).split(",")]

print(f"      Tweets with at least one response: {len(response_map):,}")

# ── CELL 6b ── Thread walker ───────────────────────────────────────────────────
def reconstruct_thread(root_id, lookup, resp_map, max_depth=10):
    thread, visited = [], set()
    current_id, depth = root_id, 0
    while current_id and depth < max_depth:
        if current_id in visited:
            break
        if current_id not in lookup:
            break
        row = lookup[current_id].copy()
        row["tweet_id"]   = current_id
        row["turn_index"] = depth
        thread.append(row)
        visited.add(current_id)
        next_ids   = resp_map.get(current_id, [])
        current_id = next_ids[0] if next_ids else None
        depth     += 1
    return thread

# ── CELL 6c ── Reconstruct all threads ────────────────────────────────────────
print("\n[5/7] Reconstructing conversation threads ...")

root_candidates = spotify_df[
    (spotify_df["inbound"] == True) &
    (
        spotify_df["in_response_to_tweet_id"].isna() |
        ~spotify_df["in_response_to_tweet_id"].isin(spotify_df["tweet_id"])
    )
]["tweet_id"].tolist()

print(f"      Candidate root tweets: {len(root_candidates):,}")

all_threads = []
for root_id in tqdm(root_candidates, desc="      Walking threads", ncols=70):
    thread = reconstruct_thread(root_id, tweet_lookup, response_map)
    if len(thread) >= 2:
        all_threads.append(thread)

print(f"      Valid threads reconstructed: {len(all_threads):,}")

# ── CELL 7 ── Flatten to records ──────────────────────────────────────────────
print("\n[6/7] Flattening threads into DataFrame ...")

def thread_to_record(thread, spotify_id):
    turns_text          = []
    first_customer_text = None
    last_spotify_reply  = None
    for turn in thread:
        author = turn.get("author_id", "")
        text   = str(turn.get("text", "")).strip()
        if author == spotify_id:
            role               = "SPOTIFY"
            last_spotify_reply = text
        else:
            role = "CUSTOMER"
            if first_customer_text is None:
                first_customer_text = text
        turns_text.append(f"{role}: {text}")
    return {
        "conversation_id"     : thread[0]["tweet_id"],
        "num_turns"           : len(thread),
        "first_customer_text" : first_customer_text,
        "last_spotify_reply"  : last_spotify_reply,
        "conversation_text"   : " | ".join(turns_text),
        "is_resolved"         : thread[-1].get("author_id") == spotify_id,
        "created_at"          : thread[0].get("created_at"),
    }

records    = [thread_to_record(t, SPOTIFY_AUTHOR_ID) for t in all_threads]
threads_df = pd.DataFrame(records)
print(f"      threads_df shape: {threads_df.shape}")

# ── QA Gate ───────────────────────────────────────────────────────────────────
print("\n      === QA REPORT ===")
print(f"      Total conversations          : {len(threads_df):,}")
print(f"      Duplicate conversation_ids   : {threads_df['conversation_id'].duplicated().sum()}")
print(f"      Null first_customer_text     : {threads_df['first_customer_text'].isna().sum()}")
print(f"      Null last_spotify_reply      : {threads_df['last_spotify_reply'].isna().sum()}")
print(f"      Resolution rate              : {threads_df['is_resolved'].mean()*100:.1f}%")
print(f"\n      Thread length distribution:")
print(threads_df["num_turns"].describe().to_string())

# ── CELL 9 ── Clean ───────────────────────────────────────────────────────────
def clean_tweet_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

threads_df["first_customer_text_clean"] = threads_df["first_customer_text"].apply(clean_tweet_text)
threads_df["last_spotify_reply_clean"]  = threads_df["last_spotify_reply"].apply(clean_tweet_text)
threads_df["customer_word_count"]       = threads_df["first_customer_text_clean"].str.split().str.len()

clean_df = (
    threads_df[threads_df["customer_word_count"] >= 4]
    .drop_duplicates(subset="first_customer_text_clean")
    .copy()
)

print(f"\n      Before cleaning : {len(threads_df):,} rows")
print(f"      After cleaning  : {len(clean_df):,} rows")
print(f"      Dropped         : {len(threads_df) - len(clean_df):,} rows")

# ── CELL 8 ── Plot (saved, not shown) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(clean_df["num_turns"], bins=20, color="#1DB954", edgecolor="black")
axes[0].set_title("Thread Length Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Number of Turns")
axes[0].set_ylabel("Count")
median_val = clean_df["num_turns"].median()
axes[0].axvline(median_val, color="red", linestyle="--", label=f"Median: {median_val}")
axes[0].legend()

resolved_counts = clean_df["is_resolved"].value_counts()
axes[1].pie(
    resolved_counts.values,
    labels=["Resolved (Spotify last)", "Unresolved (Customer last)"],
    colors=["#1DB954", "#FF4444"],
    autopct="%1.1f%%",
    startangle=90,
)
axes[1].set_title("Thread Resolution Status", fontsize=14, fontweight="bold")

plt.tight_layout()
plot_path = REPORTS / "01_thread_stats.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"      Plot saved -> {plot_path}")

# ── CELL 10 ── Save ───────────────────────────────────────────────────────────
print("\n[7/7] Saving outputs ...")

full_out = OUTPUT_PATH / "spotify_threads.csv"
clean_df.to_csv(full_out, index=False)
print(f"      Full threads CSV  : {full_out}  ({full_out.stat().st_size/1e6:.1f} MB, {len(clean_df):,} rows)")

msg_out = OUTPUT_PATH / "spotify_customer_messages.csv"
clean_df[[
    "conversation_id",
    "first_customer_text_clean",
    "num_turns",
    "is_resolved",
    "customer_word_count",
]].to_csv(msg_out, index=False)
print(f"      Customer msgs CSV : {msg_out}  ({msg_out.stat().st_size/1e6:.2f} MB, {len(clean_df):,} rows)")

# ── Sample outputs ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1 COMPLETE — Sample conversations")
print("=" * 60)
sample = clean_df[["conversation_id","num_turns","is_resolved","first_customer_text_clean"]].head(5)
for _, row in sample.iterrows():
    status = "[resolved]" if row["is_resolved"] else "[open]"
    print(f"\n  [{row['conversation_id']}]  turns={row['num_turns']}  {status}")
    print(f"  CUSTOMER: {row['first_customer_text_clean'][:120]}")

print("\n" + "=" * 60)
print(f"  CONVERSATIONS READY FOR PHASE 2: {len(clean_df):,}")
print("=" * 60)
