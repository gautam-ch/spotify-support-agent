# Engineering & Evaluation Report: Autonomous Customer Support Agent for Spotify

**Author:** Gautam Chouhan  
**Project:** Autonomous Customer Support Agent (Spotify Twitter Dataset)  
**Target Domain:** Twitter Customer Support (`@SpotifyCares`)  
**Evaluation Artifacts:** [Comparison CSV](03_evaluation_comparison.csv) | [Summary TXT](03_evaluation_summary.txt) | [Failure Analysis](03_top_failure_modes.csv)

---

## 1. Problem Framing & System Objectives

Customer support teams at scale operate under an inherent operational tension between **resolution velocity** and **support quality**. For a brand like Spotify with over 500 million active users, social media channels like Twitter (`@SpotifyCares`) receive thousands of inbound requests every hour. These range from trivial banter and feature suggestions to urgent security breaches (e.g., account hijackings) and double billings.

### 1.1 The Operational Trade-Off: Auto-Resolution vs. Escalation
In automated customer operations, AI agents create two asymmetric risks:
- **False Positives (Unnecessary Escalations):** An AI agent escalates a routine how-to question (e.g., *"How do I clear offline cache?"*) to a human specialist.  
  *Operational Impact:* Adds cost and inflates queue wait times, but preserves user trust.
- **False Negatives (Missed Escalations):** An AI agent attempts to auto-resolve an issue that strictly requires human authority (e.g., attempting to troubleshoot an unauthorized credit card charge with generic text advice).  
  *Operational Impact:* Severe user churn, financial liability, and brand degradation.

**Core System Objective:** *The agent must maximize autonomous self-serve resolution on low-risk inquiries (app bugs, feature suggestions, general FAQs) while enforcing near-zero false negatives on security, account takeover, and billing disputes through auditable, transparent escalation reasoning.*

### 1.2 Explicit Scope Boundaries: What We Chose NOT to Build
To deliver a hardened, production-grade system within the assignment scope, we explicitly bounded the architecture and chose *not* to build the following:
1. **No Direct Backend Account Mutations:** The agent does not execute refunds, password resets, or plan changes directly via internal APIs. In real-world operations, granting automated LLMs write access to customer financial and identity records without human-in-the-loop review introduces catastrophic prompt injection and unauthorized transaction risks.
2. **No Multi-Turn Stateful Session Management:** We scoped the agent to single-turn incoming message evaluation and initial resolution routing. In Twitter customer support, the critical operational bottleneck is the initial triage and routing decision; downstream conversational handling moves into private Zendesk tickets or Direct Messages.
3. **No Voice / Telephony Synthesis:** We focused exclusively on short-form text (<280 characters) matching Spotify's Twitter voice, excluding phone transcripts or multi-modal audio files.
4. **No Real-Time Multilingual Translation Layer:** While Spotify handles multi-lingual queries globally, we restricted the primary benchmark to English tweets to establish a statistically rigorous baseline without conflating translation artifacts with classification performance.

---

## 2. Data Engineering & Dialogue Reconstruction

### 2.1 Dataset Ingestion & Filtering
The raw corpus consists of the Kaggle Customer Support on Twitter dataset (`thoughtvector/customer-support-on-twitter`, 2.8M rows). Because support policies differ fundamentally across verticals (e.g., airlines handling flight delays vs. streaming services handling playback), we isolated Spotify interactions:
- Filtered specifically for `@SpotifyCares` (author ID `115888` / `SpotifyCares`).
- Discarded multi-brand mentions and non-English tweets to establish a clean benchmark.
- Resulted in **28,154 verified multi-turn Spotify customer threads**.

### 2.2 Reconstructing Inbound & Outbound Interactions
Single tweets in isolation lack conversational context. Using `in_reply_to_tweet_id` parent pointer graph traversal, we reconstructed complete conversational threads into:
1. `first_customer_text_clean`: The opening customer problem statement (e.g., *"Why am I getting charged twice for Family Premium?"*).
2. `last_spotify_reply_clean`: Spotify's verified human agent resolution (e.g., *"Hey! Send us a DM with your account email so we can inspect backstage."*).
3. `is_resolved`: Whether the thread terminated in an active resolution.

Text cleaning stripped noisy Twitter handles and normalized whitespace while retaining emojis, question marks, and capitalization necessary for sentiment and urgency detection.

---

## 3. Unsupervised Intent Discovery & Audited Golden Set

### 3.1 Embedding & Geometry Optimization
1. Sampled a representative set of 3,000 customer messages across the temporal distribution.
2. Embedded all texts into a dense 384-dimensional semantic space using `sentence-transformers/all-MiniLM-L6-v2`.
3. Applied L2-normalization to project vectors onto a unit hypersphere, making Euclidean distance in K-Means mathematically equivalent to cosine distance.

### 3.2 Elbow Method & Silhouette Validation
To select the optimal number of clusters $k$, we swept $k \in [3, 12]$ computing both inertia (Within-Cluster Sum of Squares) and Silhouette coefficients:

- [View Elbow and Silhouette Analysis Plot](02a_elbow_silhouette.png)

- **Mathematical vs Operational Optimum:** While $k=3$ produced the highest numerical Silhouette score (0.084), it conflated completely distinct operational domains (e.g., billing disputes and account hacks) into a coarse "problems" bucket.
- **Operational Choice ($k=7$):** While short Twitter text inherently produces modest silhouette scores (0.038–0.084) due to lexical noise and brevity, $k=7$ separated actionable business intents:

| Cluster | Intent Name | Volume (%) | Policy | Description & Resolution Pattern |
|:---:|---|:---:|:---:|---|
| 0 | `subscription_issue` | 18.2% | Mixed | Plan upgrades, student discount verification, family invites |
| 1 | `account_access` | 14.1% | **Escalate** | Hacked accounts, forgotten credentials, lost Facebook auth |
| 2 | `general_inquiry` | 21.4% | Auto | Country availability, catalog questions, general usage |
| 3 | `off_topic_chatter` | 11.3% | Auto | Social chatter, banter, artist praise, memes |
| 4 | `app_bug` | 16.5% | Auto | App crashes, playback stutter, offline download failures |
| 5 | `billing_dispute` | 10.8% | **Escalate** | Unauthorized charges, double billing, refund requests |
| 6 | `feature_request` | 7.7% | Auto | UI suggestions, shuffle button feedback, queue shortcuts |

- [View 2D PCA Cluster Representation Plot](02b_clusters_pca.png)

### 3.3 Golden Set Audit & Sanitization (210 Examples)
A critical requirement of this project is ensuring that the evaluation benchmark is truthful and robust. 
- **The Pitfall of Naive Regex Labeling:** Automated clustering and naive keyword rules can contaminate ground truth. For example, in an un-audited set, a tweet like *"all of my playlists disappeared... randomly plays music... connected to another device"* is easily mislabeled as `app_bug` because the word "hack" is absent.
- **The Audit:** We conducted a systematic human audit across all 210 Golden Set examples ([`data/processed/golden_set_labeled.csv`](../data/processed/golden_set_labeled.csv)):
  - Reclassified subtle account compromises into `account_access` with `is_escalation: True`.
  - Reclassified playback stutter and deleted downloads from `feature_request` into `app_bug`.
  - Corrected currency edge cases (e.g., *"charged 99kr instead of 9"*) into `billing_dispute` with `is_escalation: True`.
  - Resulted in **74 Escalations (35.2%)** and **136 Auto-handled (64.8%)** cases with detailed human rationale notes.

---

## 4. Agent Architecture: Grounded RAG Generation

### 4.1 The Hallucination & Brand Drift Problem
Standard zero-shot LLM support agents fail when prompted with generic instructions like *"You are Spotify support."* Such systems frequently:
1. Hallucinate refund timelines or invent nonexistent customer service phone numbers.
2. Generate verbose, corporate paragraphs that violate Twitter's character constraints (<280 characters).
3. Publicly request sensitive user data (passwords, payment credentials) rather than routing users into private Direct Messages (DMs).

### 4.2 Retrieval-Augmented Generation (RAG) on Brand History
Our agent pairs semantic retrieval with structured generation:
1. **Dense Semantic Retrieval:** Indexes 3,000 historical customer queries paired with verified Spotify resolutions using `all-MiniLM-L6-v2`. Computes cosine similarity in <5 ms via normalized dot products.
2. **Historical Link Sanitization:** Automatically scrubs legacy, broken shortlinks (`spoti.fi/*`, `bit.ly/*`) from historical responses, replacing them with canonical modern endpoints (`https://support.spotify.com`).
3. **Structured Pydantic Contract:** The agent emits a strictly validated schema ([`src/agent.py`](../src/agent.py#L34-L41)):
   - `reasoning`: Chain-of-thought analysis explaining intent and escalation determination.
   - `predicted_intent`: Discovered intent category.
   - `is_escalation`: Boolean decision.
   - `escalation_reason`: Auditable justification.
   - `draft_reply`: Customer-facing response grounded in historical resolutions (<60 words).

---

## 5. Escalation Decisioning & Stated Reasoning

The system implements explicit escalation contracts:
- **Mandatory Escalations:**
  - `billing_dispute` & payment transaction failures (monetary risk).
  - `account_access` & account compromises (identity & security risk).
  - Account lockouts requiring backstage internal tools.
- **Auto-Handling:**
  - Troubleshooting guides for app bugs (clean reinstall, cache wipe, OS update).
  - Feature feedback logging.
  - Policy & catalog inquiries.

**Example Stated Reasoning Output:**
> *Customer:* "@SpotifyCares someone changed my login email and I see songs streaming from another country!"  
> *Decision:* `is_escalation: True`  
> *Stated Reason:* "Account compromise and unauthorized credential alteration detected. Requires private identity verification and backstage security freeze via human specialist."  
> *Draft Reply:* "Hey! We take account security very seriously. Please send us a DM with your account's original email address and username so our security team can step in immediately."

---

## 6. Comprehensive 3-System Benchmark & Evaluation

To rigorously test our agent, we benchmarked 3 distinct systems against the audited 210-example Golden Set and a 30-example independent Human QA Benchmark:

1. **Baseline 1 (Trivial):** Majority-class predictor (`general_inquiry`), never escalates, outputs static canned template.
2. **Baseline 2 (Simple TF-IDF):** TF-IDF cosine similarity nearest neighbor for intent and historical reply retrieval + keyword matching for escalation.
3. **Proposed System (RAG Support Agent):** MiniLM semantic retrieval + calibrated classifier + Gemini structured reasoning with URL sanitization.

### 6.1 Benchmark Results Summary Table

| Metric | Baseline 1 (Trivial) | Baseline 2 (Simple TF-IDF) | Proposed RAG Agent |
|---|:---:|:---:|:---:|
| **Intent Classification Accuracy (N=210)** | 13.8% | 96.7% | **97.1%** |
| **Escalation Precision (N=210)** | 0.0% | 100.0% | **95.8%** |
| **Escalation Recall (N=210)** | 0.0% | 21.6% | **91.9%** |
| **Escalation F1-Score (N=210)** | 0.0% | 35.6% | **93.8%** |
| **Judge Fully-Correct Reply (N=30)** | 0.0% | 53.3% | **83.3%** |
| **Judge Acceptable+ Reply (N=30)** | 0.0% | 76.7% | **96.7%** |
| **Tone Acceptable+ (N=30)** | 0.0% | 73.3% | **100.0%** |

### 6.2 Key Operational Insights
- **Baseline 2 Fails on Escalation Recall:** Baseline 2 achieves 100% precision because it only escalates when exact keywords ("hacked", "stolen", "refund") are typed. However, its **recall is catastrophic at 21.6%**—meaning it misses almost 80% of real escalation cases where customers use conversational phrasing (*"my subscription says Free but bank took money"*, *"device connected to another user"*).
- **RAG Agent Achieves Production Balance:** The proposed RAG agent achieves **95.8% precision** and **91.9% recall** (F1: **93.8%**), catching subtle compromises and edge-case currency disputes while preventing queue overflow.

---

## 7. Proving Trustworthiness: Truly Independent Judge-Human Agreement

A central risk in LLM-as-a-Judge setups is **circular validation** (e.g., using heuristic rules derived from the judge's own output to claim "human agreement").

### 7.1 Methodology: Decoupled Human QA Benchmark
To establish genuine proof of trustworthiness:
1. We created a dedicated, independent benchmark file ([`data/processed/human_qa_benchmark.csv`](../data/processed/human_qa_benchmark.csv)) containing **30 representative customer tweets** across all 7 intents and complex edge cases.
2. A human support supervisor independently evaluated and pre-annotated the ground truth for each case:
   - Expected escalation decision (`correct_escalation` vs `correct_no_escalation`).
   - Target accuracy and brand tone standards.
   - Supervisor reasoning notes.
3. The automated LLM Judge was executed blind against this benchmark, and its categorical decisions were scored directly against the pre-recorded human ratings.

### 7.2 Inter-Rater Agreement Results
- **Sample Size ($N$):** 30 independent examples
- **Raw Categorical Agreement Rate:** **83.3%**
- **Cohen's Kappa ($\kappa$):** **0.595**
- **Statistical Interpretation:** $\kappa = 0.595$ represents **moderate-to-substantial inter-rater reliability** (bordering the substantial 0.60 threshold). Unlike synthetic heuristic experiments that produce inflated or arbitrary numbers, this metric reflects genuine alignment between automated LLM grading and human supervisory standards.

---

## 8. Top 5 Failure Modes & Root Cause Analysis

Auditing error traces in [`reports/03_top_failure_modes.csv`](03_top_failure_modes.csv) revealed 5 recurring failure patterns:

1. **Multi-Intent Polysemy:**  
   *Example:* *"App crashed during update and now my subscription says Free!"*  
   *Root Cause:* The customer reports both an `app_bug` and a `subscription_issue`. Single-label classification forces an either/or choice.
2. **Subtle Compromise Signals:**  
   *Example:* *"Why is Russian rap playing on my account when I listen to jazz?"*  
   *Root Cause:* The user never typed "hacked" or "password". While semantic retrieval surfaces compromise cases, low-confidence thresholds can occasionally route these as audio bugs.
3. **Historical Link Invalidation:**  
   *Example:* Historical Spotify replies from 2017 referencing `spoti.fi/reinstall` shortlinks that are now broken.  
   *Solution Implemented:* Our retriever's URL sanitizer intercepts and transforms legacy shortlinks into canonical `https://support.spotify.com` endpoints.
4. **Regional Telecom Billing Nuances:**  
   *Example:* Payment failures via third-party telecom carrier billing (e.g., *"not able to pay 9 pesos through smart"*).  
   *Root Cause:* Sparse representation in the global English dataset (<1% of corpus), leading to lower classifier confidence.
5. **Sarcasm & Passive-Aggressive Tone:**  
   *Example:* *"Oh brilliant job Spotify, love when my downloaded music disappears on a flight."*  
   *Root Cause:* Surface lexical tokens contain positive words (*"brilliant"*, *"love"*), which distort simple sentiment filters unless chain-of-thought reasoning evaluates the underlying complaint.

---

## 9. What is Misleading About My Headline Number? (Honesty Section)

High headline numbers (e.g., *97.1% intent accuracy* and *93.8% escalation F1*) can breed false complacency. Intellectual honesty requires detailing where these metrics are fragile:

1. **Same-Family Model Preference Bias:** Both the Agent and Judge use Google Gemini models (`gemini-3.5-flash-lite`). LLM evaluators inherently share token generation likelihoods and stylistic biases with models of the same family. A cross-model evaluator (e.g., Anthropic Claude 3.5 Sonnet or GPT-4o) would likely score reply tone and accuracy more stringently.
2. **Exclusion of Non-English & Code-Mixed Tweets:** The dataset filters for English text. In production, `@SpotifyCares` handles massive volumes in Spanish, Portuguese, Indonesian, and Taglish. Benchmark performance will drop on code-mixed queries.
3. **Twitter Brevity Bias:** Twitter support interactions average under 25 words. Short, concise queries are easier for dense vector models to classify. These metrics will not directly translate to multi-paragraph email tickets containing stacked issues and attachments.
4. **Stratified vs In-The-Wild Class Distribution:** The evaluation set artificially balances all 7 intents (~30 per class). In real-world production, traffic is heavily skewed: during major outages, `app_bug` represents >75% of volume; during billing cycles, `billing_dispute` spikes.

---

## 10. What I Would Do With One More Week (Technical Roadmap)

If allocated one additional week to advance this system toward Tier-1 production readiness, I would implement:

1. **LoRA Fine-Tuning of a Compact Open-Source LLM (Llama-3-8B / Qwen-2.5-7B):**
   - *Objective:* Replace commercial API calls with a self-hosted, fine-tuned 8B model trained on the 28,154 verified `@SpotifyCares` dialogue threads.
   - *Impact:* Cuts per-ticket inference latency from 1,200 ms to <150 ms and eliminates third-party API costs.
2. **Redis-Backed Semantic FAQ Caching:**
   - *Objective:* Implement an exact- and near-cosine semantic cache on Redis for top-50 high-frequency FAQs (e.g., *"how to cancel premium"*, *"offline playlist limits"*).
   - *Impact:* Serves ~40% of routine traffic sub-10 ms with zero LLM compute overhead.
3. **Human-in-the-Loop Active Learning Queue:**
   - *Objective:* Build an automated queue routing tickets where classifier confidence is below 0.65 or escalation reason contains ambiguous markers directly to human agents.
   - *Impact:* Human agent corrections are logged back into the training corpus, establishing a continuous learning loop.
4. **Multilingual Embedding & Canonical URL Registry:**
   - *Objective:* Upgrade vector embeddings to `paraphrase-multilingual-MiniLM-L12-v2` and build a live crawler that verifies all surfaced Spotify support URLs against active HTTP 200 endpoints.

---

## 11. Decision Log: 12 Non-Obvious Engineering Decisions

| # | Decision Made | Alternative Rejected | Rationale & Operational Trade-Off |
|---|---|---|---|
| **1** | **Filter exclusively for `@SpotifyCares`** | Training a multi-brand cross-industry model | Customer support tone and resolution protocols are domain-specific. Airline policies (baggage/flight delays) dilute streaming service policies (DRM/playback/family plans). |
| **2** | **L2 Normalize MiniLM Embeddings** | Unnormalized Euclidean distance | Short tweets have smaller embedding norms than verbose tweets. L2 normalization ensures K-Means distance reflects pure semantic direction rather than message length. |
| **3** | **Select $k=7$ over mathematical optimum $k=3$** | Retaining $k=3$ with highest Silhouette score (0.084) | $k=3$ grouped account hacks and billing disputes into a single "trouble" cluster. $k=7$ cleanly isolates critical business routing boundaries despite lower silhouette values. |
| **4** | **Audited Golden Set over Pure Heuristic Labeling** | Automated regex keyword labeling | Automated regex scripts miss euphemisms (e.g., account hijack without typing "hack"). Manual audit ensures ground truth integrity. |
| **5** | **Decoupled 30-Example Human QA Benchmark** | Using LLM Judge self-predictions for Kappa calculation | Circular validation invalidates trust. Pre-annotated independent human ratings provide statistically valid Cohen's Kappa measurement. |
| **6** | **Dual-Candidate Side-by-Side Judging Prompt** | Evaluating Candidate A and B in separate API calls | Cuts evaluation API costs and quota consumption by 50% while providing the judge direct comparative context. |
| **7** | **Chain-of-Thought Before Structured Scoring** | Outputting categorical grades directly | Forcing the LLM to output natural language reasoning first improves categorical scoring consistency and reduces classification hallucinations. |
| **8** | **Strict Escalation Recall Priority over Precision** | Maximizing overall F1 equally | In customer support, a false negative (missed security breach) causes customer churn and liability; a false positive merely adds a ticket to a human queue. |
| **9** | **Historical URL Sanitization Layer** | Permitting historical replies to emit original links | 2017 training data contains dead `spoti.fi` shortlinks. Replacing with `https://support.spotify.com` prevents broken customer experiences. |
| **10** | **Local Embedding & Scikit-Learn Classifier** | Pure Zero-Shot LLM Intent Prompting | Local inference executes in <10 ms on CPU at zero API cost; provides calibrated probabilities for fallback thresholding. |
| **11** | **Exponential Backoff with Header-Aware Sleep** | Fixed-delay retry loops | Google GenAI free tier imposes strict RPM/RPD limits. Parsing retry delay recommendations avoids permanent HTTP 429 locks. |
| **12** | **Disk Caching for Evaluation Runs (`eval_cache.json`)** | In-memory evaluation arrays | Guarantees that evaluation benchmarks are resilient to transient network drops and quota throttling without re-spending credits. |

---

## 12. Conclusion & Production Readiness

By pairing **dense semantic retrieval on verified brand history** with **structured escalation reasoning** and an **audited evaluation harness**, this project proves that AI customer support can move beyond brittle keyword bots without risking ungrounded hallucinations.

With an **Escalation F1-score of 93.8%**, **97.1% intent classification accuracy**, and an independently validated **0.595 Cohen's Kappa**, the system provides the mathematical proof and operational transparency required for live deployment.
