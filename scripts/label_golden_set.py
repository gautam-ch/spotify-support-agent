"""
scripts/label_golden_set.py - Semi-Automated Ground Truth Annotator for Golden Set

Labels `is_escalation` and `label_notes` for the 210 golden set examples
based on customer support domain rules, producing `golden_set_labeled.csv`.

Rules for Escalation (is_escalation = True):
1. Security & Account Hijack: Account hacked, credentials altered, unauthorized access.
2. Account Recovery: Locked out, password reset email not received, deleted social login.
3. Financial / Billing: Unauthorized charges, double billing, refund requests, billing errors.
4. Subscription Verification: Student / Family plan verification failures requiring internal tool access.

Rules for Auto-Handle (is_escalation = False):
1. App Troubleshooting: Cache clearing, reinstallation, playback stutter, offline download advice.
2. Feature Requests: UI feedback, shuffle button size, feature wishlists.
3. General Inquiries: Availability in countries, pricing questions, how-to usage.
4. Off-Topic Chatter: Casual banter, jokes, song reactions.
"""
import sys
import re
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INPUT_PATH = Path("data/processed/golden_set_unlabeled_v2.csv")
OUTPUT_PATH = Path("data/processed/golden_set_labeled.csv")


def determine_escalation(row) -> tuple[bool, str]:
    text = str(row.get("first_customer_text_clean", "")).lower()
    intent = str(row.get("intent_label", "")).lower()

    # Rule 1: High-risk security keywords -> ESCALATE
    security_patterns = [
        r"\bhack(ed)?\b", r"\bcompromis(ed)?\b", r"\bstolen\b",
        r"\bchanged my (email|password)\b", r"\bwithout my (permission|consent)\b",
        r"\bunauthorized\b", r"\bsuspicious\b"
    ]
    for pat in security_patterns:
        if re.search(pat, text):
            return True, "Security/Hijack: Account compromise requires human verification."

    # Rule 2: Account Access / Lockout -> ESCALATE
    account_lock_patterns = [
        r"\bcant (log|sign) in\b", r"\bcan't (log|sign) in\b", r"\bcannot (log|sign) in\b",
        r"\bunable to (log|sign) in\b", r"\bdeleted my (facebook|fb)\b",
        r"\bno longer have access\b", r"\bpassword reset\b", r"\blocked out\b",
        r"\bcheck your dms?\b", r"\bsent (you )?a dm\b"
    ]
    for pat in account_lock_patterns:
        if re.search(pat, text):
            return True, "Account Access: User cannot access account / needs DM identity verification."

    # Rule 3: Billing Dispute / Financial -> ESCALATE
    billing_patterns = [
        r"\bcharged twice\b", r"\bcharged 2 times\b", r"\bdouble charge\b",
        r"\brefund\b", r"\bdispute\b", r"\bwrong (amount|charge)\b",
        r"\bpayment (failed|problem|issue)\b", r"\bnot able to pay\b",
        r"\bcancel (my )?(premium|subscription)\b", r"\bcharged but\b",
        r"\bcharged .* not free\b", r"\bcharged multiple\b"
    ]
    for pat in billing_patterns:
        if re.search(pat, text):
            return True, "Billing Dispute: Monetary transactions and refund requests require human review."

    # Rule 4: Intent-specific default fallbacks
    if intent == "billing_dispute":
        return True, "Billing Dispute: Financial issue requiring human account inspection."
    if intent == "account_access":
        # Check if it's just a general how-to vs actual lockout
        if any(w in text for w in ["how do i", "is it possible", "how to"]):
            return False, "General question about account settings (self-serve)."
        return True, "Account Access: Sign-in or credential roadblock requiring backstage check."

    # Rule 5: Feature requests, off-topic, app bugs, general inquiry -> AUTO-HANDLE
    if intent == "feature_request":
        return False, "Feature Request: Feedback or enhancement suggestion (auto-handled)."
    if intent == "off_topic_chatter":
        return False, "Off-topic banter or social engagement (auto-handled)."
    if intent == "general_inquiry":
        return False, "General Inquiry: FAQ or product question resolvable via standard info."
    if intent == "app_bug":
        return False, "Technical Bug: Resolvable via standard self-serve troubleshooting (cache/reinstall)."
    if intent == "subscription_issue":
        # Check if money/charge is involved
        if any(w in text for w in ["pay", "card", "bill", "charge", "promo", "code", "family", "plan"]):
            return True, "Subscription/Payment: Plan setup or billing issue requiring human check."
        return False, "Subscription information resolvable via FAQ."

    return False, "Default: Standard customer inquiry."


def main():
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} golden set candidates from {INPUT_PATH}")

    escalations = []
    notes = []

    for _, row in df.iterrows():
        is_esc, note = determine_escalation(row)
        escalations.append(is_esc)
        notes.append(note)

    df["is_escalation"] = escalations
    df["label_notes"] = notes

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSuccessfully labeled {len(df)} examples and saved to:")
    print(f"  -> {OUTPUT_PATH}")
    print(f"  -> {INPUT_PATH}")

    print("\nEscalation distribution:")
    print(df["is_escalation"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
    print(df["is_escalation"].value_counts())

    print("\nEscalation breakdown by Intent:")
    ct = pd.crosstab(df["intent_label"], df["is_escalation"], margins=True)
    print(ct)


if __name__ == "__main__":
    main()
