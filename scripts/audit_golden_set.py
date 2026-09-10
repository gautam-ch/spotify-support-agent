"""
scripts/audit_golden_set.py - Audit & Correction of the Golden Set (210 examples)

Inspects every row for known misclassification patterns (e.g., subtle account
compromise labelled as app_bug, regional billing labelled as off_topic) and
applies corrections.

IMPORTANT: After running this script, the golden set MUST be manually reviewed
by a human annotator. This script handles programmatic corrections for
clear-cut cases only. Ensure full manual review of the annotations.
"""
import sys
import re
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOLDEN_PATH = Path("data/processed/golden_set_labeled.csv")
HUMAN_BENCHMARK_PATH = Path("data/processed/human_qa_benchmark.csv")


def audit_and_correct():
    df = pd.read_csv(GOLDEN_PATH)
    print(f"Loaded {len(df)} golden set examples.")

    corrections_count = 0

    for idx, row in df.iterrows():
        text = str(row["first_customer_text_clean"])
        text_lower = text.lower()
        old_intent = row["intent_label"]
        old_esc = row["is_escalation"]
        new_intent = old_intent
        new_esc = old_esc
        new_notes = row["label_notes"]

        # 1. Check for account compromise / hijack disguised as app_bug or chatter
        if ("randomly plays music" in text_lower and "another device" in text_lower) or \
           ("sy mau lapor akun yang kena hack" in text_lower) or \
           ("someone logged into my account" in text_lower) or \
           ("compromised" in text_lower) or \
           ("stolen" in text_lower) or \
           ("my account has been hacked" in text_lower):
            new_intent = "account_access"
            new_esc = True
            new_notes = "Security: Account compromise or suspicious unauthorized device activity requires human intervention."

        # 2. Check for billing disputes misclassified under chatter or feature requests
        elif ("charged me 99kr instead of 9" in text_lower) or \
             ("charged twice" in text_lower) or \
             ("refund" in text_lower) or \
             ("double charge" in text_lower) or \
             ("payment failure" in text_lower and "bank" in text_lower) or \
             ("not able to pay 9 pesos" in text_lower):
            if new_intent in ["off_topic_chatter", "feature_request", "general_inquiry"]:
                new_intent = "billing_dispute"
            new_esc = True
            new_notes = "Financial: Monetary charge discrepancy, billing failure, or refund request requiring account verification."

        # 3. Check for app bugs misclassified under feature requests
        elif ("skip and stutter" in text_lower) or \
             ("why u keep deleting" in text_lower and "download" in text_lower) or \
             ("lost control centre integration" in text_lower) or \
             ("app keep crashing" in text_lower) or \
             ("crash" in text_lower and "update" in text_lower):
            if new_intent == "feature_request":
                new_intent = "app_bug"
                new_esc = False
                new_notes = "App Bug: Playback stutter, crash, or local download glitch resolvable via self-serve troubleshooting."

        # 4. Check for service status / outage inquiries misclassified as off-topic chatter
        elif ("problems with streaming at the moment" in text_lower) or \
             ("is spotify down" in text_lower):
            new_intent = "app_bug"
            new_esc = False
            new_notes = "Service Status: Inquiring about current streaming availability or temporary service interruption."

        # 5. Check subscription vs billing distinction
        elif old_intent == "subscription_issue":
            if "i am not able to sign up" in text_lower or "how to join" in text_lower:
                new_esc = False
                new_notes = "Subscription FAQ: How-to guidance on signing up or plan features (auto-handled)."
            elif any(w in text_lower for w in ["charged", "refund", "twice", "bill", "smart", "pesos", "card", "pay"]):
                new_esc = True
                new_notes = "Subscription Billing: Account-specific payment issue or family plan billing failure."

        # 6. Check feature request vs general feedback
        elif old_intent == "feature_request":
            new_esc = False
            new_notes = "Feature Request: UI suggestions, wishlist items, or playlist mechanics."

        # 7. Check general inquiry
        elif old_intent == "general_inquiry":
            if any(w in text_lower for w in ["hack", "login", "password"]):
                new_intent = "account_access"
                new_esc = True
                new_notes = "Account Access: Security/sign-in issue."
            else:
                new_esc = False
                new_notes = "General Inquiry: Public information, catalog, or channel availability."

        # 8. Check off topic chatter
        elif old_intent == "off_topic_chatter":
            new_esc = False
            new_notes = "Off-Topic: Casual social interaction, memes, or artist praise."

        if new_intent != old_intent or new_esc != old_esc:
            corrections_count += 1
            print(f"Corrected [{idx}]: {old_intent} (esc={old_esc}) -> {new_intent} (esc={new_esc}) | Text: {text[:60]}")

        df.at[idx, "intent_label"] = new_intent
        df.at[idx, "is_escalation"] = bool(new_esc)
        df.at[idx, "label_notes"] = new_notes

    print(f"\nTotal corrections applied: {corrections_count}")
    df.to_csv(GOLDEN_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved audited Golden Set to {GOLDEN_PATH}")
    print("\nAudited Intent distribution:")
    print(df["intent_label"].value_counts())
    print("\nAudited Escalation distribution:")
    print(df["is_escalation"].value_counts())

    # ── Create Human QA Benchmark TEMPLATE (30 items) ─────────────────────────
    # This creates a blank template for manual human annotation.
    # The annotator MUST fill in the human_* columns by reading each tweet
    # and making independent judgments.
    benchmark_indices = []
    for intent in df["intent_label"].unique():
        sub = df[df["intent_label"] == intent]
        n_pick = 4 if intent in ["account_access", "billing_dispute"] else 4
        benchmark_indices.extend(sub.index[:n_pick].tolist())

    if len(benchmark_indices) < 30:
        remaining = [i for i in df.index if i not in benchmark_indices]
        benchmark_indices.extend(remaining[: (30 - len(benchmark_indices))])
    benchmark_indices = benchmark_indices[:30]

    bench_df = df.loc[benchmark_indices].copy().reset_index(drop=True)

    # Leave human annotation columns BLANK for manual filling
    bench_df["human_expected_escalation"] = ""   # Fill: correct_escalation / correct_no_escalation / missed_escalation / unnecessary_escalation
    bench_df["human_target_accuracy"] = ""        # Fill: fully_correct / partially_correct / incorrect
    bench_df["human_target_tone"] = ""            # Fill: excellent / acceptable / poor
    bench_df["human_qa_notes"] = ""               # Fill: Your reasoning for each judgment

    bench_df.to_csv(HUMAN_BENCHMARK_PATH, index=False, encoding="utf-8-sig")
    print(f"\nGenerated Human QA Benchmark TEMPLATE ({len(bench_df)} rows) at {HUMAN_BENCHMARK_PATH}")
    print("⚠️  IMPORTANT: You must manually fill in the human_* columns before running phase3_eval.py")
    print("   Ensure manual annotations are complete.")


if __name__ == "__main__":
    audit_and_correct()
