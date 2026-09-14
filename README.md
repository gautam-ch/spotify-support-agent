# AI Customer Support Agent (Spotify Twitter Dataset)


## 📌 Executive Summary

Modern automated customer support systems often fail because they either rely on rigid static templates or generate ungrounded, hallucinated LLM responses. This system solves this problem through an end-to-end grounded pipeline:

1. **Unsupervised Intent Discovery**: Discovered 7 distinct customer intents from 28,000+ real-world Spotify Twitter interactions using `sentence-transformers` embeddings, Elbow Method analysis, and Silhouette score optimization ($k=3 \dots 12$).
2. **Historical Grounding (RAG)**: Uses dense semantic retrieval over 3,000 verified historical brand interactions to ground drafted replies in Spotify's historical resolutions.
3. **Escalation Reasoning**: Categorizes incoming messages into self-serve vs. human-escalation with explicit, auditable chain-of-thought justifications.
4. **Three-System Benchmark**: Evaluated against a Trivial Baseline (majority-class) and a Simple Baseline (TF-IDF + keyword heuristics) on a 210-example stratified golden set (rule-based pre-labeling + manual human verification).
5. **LLM-as-a-Judge Validation**: Validated using Gemini with categorical rubrics (Accuracy, Tone, Escalation alignment), measured with **Cohen's Kappa** against independently hand-annotated human QA benchmark to prove trustworthiness.

---

## 🏛️ System Architecture

```
                                 [ Incoming Customer Tweet ]
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         [ Intent Classifier ]                               [ Semantic RAG Retriever ]
      Logistic Regression on MiniLM                        Dense Cosine Similarity over
      Calibrated Probabilities (7 classes)                 3,000 Historical Interactions
                    │                                                   │
                    │   Predicted Intent + Confidence                   │ Top-3 Historical Resolutions
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                                 [ Spotify Support Agent ]
                              Structured Generation (Gemini)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       [ Escalation Decision ]                                [ Grounded Reply ]
      • is_escalation: True / False                         • On-brand Twitter style (<60 words)
      • escalation_reason: Stated justification             • Grounded in verified historical actions
                                                            • Security-first DM routing when needed
```

---

## 📂 Project Structure

```
spotify-support-agent/
├── data/
│   ├── raw/
│   │   └── twcs.csv                       # Raw Kaggle customer support dataset
│   └── processed/
│       ├── spotify_threads.csv            # Reconstructed customer-brand conversation threads
│       ├── spotify_customer_messages.csv  # 28K customer opening messages
│       ├── sample_with_clusters.csv       # 3,000 clustered messages with intent labels
│       ├── embeddings_cache.npy           # Cached 384-dim MiniLM embeddings
│       ├── golden_set_unlabeled_v2.csv    # 210 stratified candidate examples
│       ├── golden_set_labeled.csv         # 210 ground-truth annotated examples
│       ├── human_qa_benchmark.csv         # 30 independently hand-annotated QA examples
├── src/
│   ├── __init__.py
│   ├── utils.py                           # Shared utilities (retry logic, etc.)
│   ├── retriever.py                       # Dense vector retriever (SentenceTransformers)
│   ├── classifier.py                      # Supervised Intent Classifier (LogisticRegression)
│   ├── agent.py                           # Structured RAG Support Agent
│   └── baselines.py                       # Baseline 1 (Trivial) and Baseline 2 (Simple TF-IDF)
├── scripts/
│   ├── phase1_run.py                      # Data filtering & conversation reconstruction
│   ├── phase2_cluster.py                  # K-Means clustering, Elbow & Silhouette analysis
│   ├── label_golden_set.py                # Rule-based pre-labeling for golden set
│   ├── audit_golden_set.py                # Correction pass & human benchmark template
│   ├── import_annotations.py              # Validate & import manual annotations
│   ├── phase3_eval.py                     # 3-System Evaluation Harness & Judge
│   └── chat.py                            # Interactive CLI for testing
├── reports/
│   ├── REPORT.md                          # Full engineering & evaluation report
│   ├── 01_thread_stats.png                # Dataset volume & turn distribution
│   ├── 02a_elbow_silhouette.png           # Elbow & Silhouette validation curves
│   ├── 02b_clusters_pca.png               # 2D PCA cluster visualization
│   ├── 03_evaluation_comparison.csv       # Detailed per-example benchmark results
│   ├── 03_top_failure_modes.csv           # Identified failure cases for root-cause analysis
│   └── 03_evaluation_summary.txt          # Complete metric summary
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart & Reproducibility Guide (<15 min)

### 1. Prerequisites & Environment Setup
Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/gautam-ch/spotify-support-agent.git
cd spotify-support-agent

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_CHAT_MODEL=gemini-3.6-flash
GEMINI_JUDGE_MODEL=gemini-3.6-flash
```

### 3. Run Pipeline Step-by-Step

#### Step 1: Process Raw Data (Optional if processed files exist)
```bash
python scripts/phase1_run.py
```
*Extracts 28,000+ Spotify conversations, cleans text, and reconstructs multi-turn dialogue threads.*

#### Step 2: Unsupervised Intent Discovery
```bash
python scripts/phase2_cluster.py
```
*Computes local MiniLM embeddings, evaluates $k=3 \dots 12$ using Silhouette scores and inertia (Elbow method), generates PCA visualizations, and stratifies the 210-example Golden Set.*

#### Step 3: Verify & Annotate Golden Set
```bash
python scripts/label_golden_set.py

# Optional: Validate hand-labeled annotations and review distributions:
python scripts/import_annotations.py
```
*Validates the 210 stratified golden set examples and ensures annotations comply with the expected schema.*

#### Step 4: Run the Complete 3-System Evaluation Benchmark
```bash
# Run benchmark against the 30-example independent Human QA Benchmark:
.venv/Scripts/python scripts/phase3_eval.py --n-eval 30

# Or run full evaluation across all 210 audited golden set rows:
.venv/Scripts/python scripts/phase3_eval.py --full
```

#### Step 5: Test Custom Messages Interactively (CLI)
```bash
# 1. Interactive terminal chat session:
.venv/Scripts/python scripts/chat.py

# 2. One-off test from command line:
.venv/Scripts/python scripts/chat.py --message "@SpotifyCares my account was hacked and songs are playing"

# 3. Compare side-by-side with Baselines & view historical retrieved matches:
.venv/Scripts/python scripts/chat.py --compare --verbose --message "@SpotifyCares I was billed twice for Family Premium"
```

---

## 🎯 Intent Taxonomy (Discovered from Data)

Through K-Means clustering and semantic inspection of centroid-adjacent tweets, 7 core intents were formalized:

| Intent Label | Description | Escalation Policy | Typical Resolution Pattern |
|---|---|---|---|
| `account_access` | Sign-in failures, hacked accounts, 2FA issues | **Escalate (True)** | DM verification of email/username backstage |
| `billing_dispute` | Double charges, unauthorized charges, refunds | **Escalate (True)** | Secure DM lookup of transaction records |
| `subscription_issue` | Premium drop, family invite errors, student discount | **Escalate (True)** if payment; **Auto (False)** if FAQ | Verification link or DM depending on billing status |
| `app_bug` | App crashes, playback stutter, offline download sync | **Auto-Handle (False)** | Clean reinstall steps, cache clear, OS version check |
| `feature_request` | UI complaints, shuffle button requests, lyrics | **Auto-Handle (False)** | Empathetic acknowledgment, log feedback for dev team |
| `general_inquiry` | International availability, plan limits, FAQs | **Auto-Handle (False)** | Direct answer with official policy information |
| `off_topic_chatter` | Memes, casual banter, song praise | **Auto-Handle (False)** | Friendly, brand-aligned casual response |

---

## 📊 Benchmark Results

Evaluated across the audited 210-example Golden Set and the 30-example independent Human QA Benchmark:

| Metric | Baseline 1 (Trivial) | Baseline 2 (Simple TF-IDF) | Proposed RAG Agent |
|---|:---:|:---:|:---:|
| **Intent Classification Accuracy (N=210)** | 13.8% | 96.7% | **97.1%** |
| **Escalation Precision (N=210)** | 0.0% | 100.0% | **95.8%** |
| **Escalation Recall (N=210)** | 0.0% | 21.6% | **91.9%** |
| **Escalation F1-Score (N=210)** | 0.0% | 35.6% | **93.8%** |
| **Judge Fully-Correct Replies (N=30)** | 0.0% | 53.3% | **83.3%** |
| **Judge Acceptable+ Replies (N=30)** | 0.0% | 76.7% | **96.7%** |
| **Judge Acceptable+ Tone (N=30)** | 0.0% | 73.3% | **100.0%** |

### Proving Trustworthiness: Truly Independent Judge-Human Agreement
To establish genuine proof of trustworthiness without circular logic:
- **Sample Size ($N$):** 30 pre-annotated customer interactions independently graded by human QA supervisor standards.
- **Raw Categorical Agreement:** **83.3%**
- **Cohen's Kappa ($\kappa$):** **0.595** (Moderate-to-Substantial Inter-Rater Reliability).
- Proves automated LLM evaluation aligns with human support supervisors on operational triage.

> [!NOTE]
> **Reproducibility & Custom Annotation Protocol**:
> The benchmark results above reflect the audited baseline dataset. Reviewers or annotators can independently hand-label any subset of rows, validate the dataset using `python scripts/import_annotations.py`, and re-run evaluation in under 3 minutes via `python scripts/phase3_eval.py --n-eval 30`.

---

## 🔍 Top 5 Failure Modes & Root Cause Analysis

1. **Multi-Intent Polysemy**: Customer reports both a billing error and an app crash in one tweet. Single-label classification forces an arbitrary choice.
2. **Subtle Compromise Signals**: User says *"my saved songs vanished and French songs are playing"* without using words like *"hacked"*. Requires dense semantic detection to prevent false negatives.
3. **Historical Link Invalidation**: Historical training data contains dead shortlinks (`spoti.fi/xxx`) that no longer resolve. Solved via a retriever URL sanitization layer replacing dead links with `https://support.spotify.com`.
4. **Localized Currency / Telecom Nuances**: Inquiries regarding regional carriers (e.g., Smart Telecom in Philippines, 99 SEK in Sweden) have sparse representation (<1%), reducing classifier confidence.
5. **Sarcastic Complaints**: Tweets like *"Thanks Spotify for deleting my playlist right before my party"* can distort basic sentiment if not analyzed with chain-of-thought reasoning.

---

## ⚠️ "What is Misleading About My Headline Number?" (Honesty Section)

1. **Same-Model Preference Bias**: The Judge and Agent both use Google Gemini models, introducing stylistic alignment bias where the Judge inherently favors the Agent's syntactic structure.
2. **English Language Exclusivity**: The pipeline filters exclusively for English tweets; performance will degrade on multilingual or code-mixed support requests.
3. **Twitter Short-Form Simplicity**: Twitter messages average <25 words. This high accuracy cannot be assumed to transfer directly to multi-paragraph email tickets or phone transcripts.
4. **Stratified vs Real-World Distribution**: The Golden Set artificially balances all 7 intents equally (~30 per class), whereas in real-world deployment, `app_bug` and `billing_dispute` spikes occur during outages.

---

## 🚫 What We Chose NOT to Build (Scope Boundaries)
- **No Direct Backend Account Mutations:** No automated refunds or credential resets via API to protect against prompt injection and monetary risk.
- **No Multi-Turn Stateful Memory:** Focused on initial triage and routing, matching Twitter support handover into Zendesk.
- **No Voice or Telephony Support:** Exclusively optimized for concise social text (<280 chars).

---

## 🚀 What I Would Do With One More Week
1. **LoRA Fine-Tuning (Llama-3-8B / Qwen-2.5):** Fine-tune open-source weights on 28K threads to drop inference latency below 150 ms and remove third-party API costs.
2. **Redis Semantic FAQ Caching:** Sub-10ms response caching for top-50 repetitive queries.
3. **Active Learning Human UI Queue:** Automatically route low-confidence (<0.65) tickets to human supervisors for active model retraining.

---

## 📜 Decision Log: 12 Non-Obvious Decisions
See [Section 11 in Engineering Report](reports/REPORT.md#11-decision-log-12-non-obvious-engineering-decisions) for the complete log of 12 engineering decisions, alternatives rejected, and operational trade-offs.

---

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Embeddings & Search**: `sentence-transformers` (`all-MiniLM-L6-v2`), Scikit-learn, NumPy
- **LLM & Structured Generation**: Google GenAI SDK (`gemini-3.5-flash-lite`), Pydantic
- **Data Engineering**: Pandas, Matplotlib, Seaborn

---

## 📚 Citations & Attributions
- **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) by ThoughtVector (Kaggle, 2.8M rows)
- **Embeddings**: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) by Sentence-Transformers (22M params, Apache 2.0)
- **LLM API**: [Google GenAI SDK](https://googleapis.github.io/python-genai/) for Gemini model access
- **ML Framework**: [Scikit-learn](https://scikit-learn.org/) (BSD-3-Clause)
- **Golden Set Labeling Methodology**: Rule-based pre-labeling with manual human verification.
