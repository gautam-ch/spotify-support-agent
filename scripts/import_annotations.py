"""
scripts/import_annotations.py - Validate and import hand-labeled annotations.

This script audits the user's manual annotations in:
1. data/processed/golden_set_labeled.csv (210 rows)
2. data/processed/human_qa_benchmark.csv (30 rows)

It verifies format, allowed category values, flags any empty or malformed
cells, and prints a summary report to ensure the dataset
is clean and ready for evaluation.
"""
import sys
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GOLDEN_PATH = Path("data/processed/golden_set_labeled.csv")
BENCHMARK_PATH = Path("data/processed/human_qa_benchmark.csv")

VALID_INTENTS = {
    "account_access",
    "billing_dispute",
    "subscription_issue",
    "app_bug",
    "feature_request",
    "general_inquiry",
    "off_topic_chatter",
}

VALID_HUMAN_ESCALATION = {
    "correct_escalation",
    "correct_no_escalation",
    "missed_escalation",
    "unnecessary_escalation",
}

VALID_HUMAN_ACCURACY = {
    "fully_correct",
    "partially_correct",
    "incorrect",
}

VALID_HUMAN_TONE = {
    "excellent",
    "acceptable",
    "poor",
}


def validate_golden_set():
    print("=" * 70)
    print("1. VALIDATING GOLDEN SET (data/processed/golden_set_labeled.csv)")
    print("=" * 70)

    if not GOLDEN_PATH.exists():
        print(f"❌ Error: {GOLDEN_PATH} does not exist.")
        return False

    df = pd.read_csv(GOLDEN_PATH)
    print(f"Found {len(df)} rows.")

    errors = []
    warnings = []

    if len(df) != 210:
        warnings.append(f"Row count is {len(df)}, expected 210.")

    required_cols = [
        "conversation_id",
        "intent_label",
        "first_customer_text_clean",
        "is_escalation",
    ]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if errors:
        for err in errors:
            print(f"  ❌ {err}")
        return False

    # Check intents
    invalid_intents = df[~df["intent_label"].isin(VALID_INTENTS)]
    if not invalid_intents.empty:
        for idx, row in invalid_intents.iterrows():
            errors.append(
                f"Row {idx+2}: Invalid intent '{row['intent_label']}'. Must be one of: {sorted(VALID_INTENTS)}"
            )

    # Check is_escalation
    non_bool = df[~df["is_escalation"].astype(str).str.lower().isin(["true", "false", "1", "0"])]
    if not non_bool.empty:
        for idx, row in non_bool.iterrows():
            errors.append(
                f"Row {idx+2}: Invalid is_escalation '{row['is_escalation']}'. Must be True or False."
            )

    # Check for empty text
    empty_text = df[df["first_customer_text_clean"].isna() | (df["first_customer_text_clean"].astype(str).str.strip() == "")]
    if not empty_text.empty:
        for idx, row in empty_text.iterrows():
            errors.append(f"Row {idx+2}: Empty customer text.")

    if errors:
        print(f"\n❌ Found {len(errors)} error(s) in golden set:")
        for err in errors[:10]:
            print(f"   {err}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more.")
        return False

    # Summary
    print("\n Intent Distribution:")
    for intent, count in df["intent_label"].value_counts().items():
        print(f"   - {intent:<22}: {count:>3} rows")

    esc_counts = df["is_escalation"].astype(bool).value_counts()
    print("\n Escalation Distribution:")
    print(f"   - Needs Human (True) : {esc_counts.get(True, 0):>3} rows")
    print(f"   - Auto-handle (False): {esc_counts.get(False, 0):>3} rows")

    print("\n✅ Golden Set validation passed!")
    return True


def validate_human_benchmark():
    print("\n" + "=" * 70)
    print("2. VALIDATING HUMAN QA BENCHMARK (data/processed/human_qa_benchmark.csv)")
    print("=" * 70)

    if not BENCHMARK_PATH.exists():
        print(f"❌ Error: {BENCHMARK_PATH} does not exist.")
        return False

    df = pd.read_csv(BENCHMARK_PATH)
    print(f"Found {len(df)} rows.")

    errors = []
    unfilled = []

    required_cols = [
        "human_expected_escalation",
        "human_target_accuracy",
        "human_target_tone",
        "human_qa_notes",
    ]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if errors:
        for err in errors:
            print(f"  ❌ {err}")
        return False

    for idx, row in df.iterrows():
        row_num = idx + 2
        text_snip = str(row.get("first_customer_text_clean", ""))[:45]

        esc = str(row.get("human_expected_escalation", "")).strip()
        acc = str(row.get("human_target_accuracy", "")).strip()
        tone = str(row.get("human_target_tone", "")).strip()
        notes = str(row.get("human_qa_notes", "")).strip()

        # Check for empty cells
        empty_fields = []
        if not esc or esc == "nan":
            empty_fields.append("human_expected_escalation")
        if not acc or acc == "nan":
            empty_fields.append("human_target_accuracy")
        if not tone or tone == "nan":
            empty_fields.append("human_target_tone")

        if empty_fields:
            unfilled.append((row_num, text_snip, empty_fields))
            continue

        # Check valid choices
        if esc not in VALID_HUMAN_ESCALATION:
            errors.append(
                f"Row {row_num}: Invalid escalation '{esc}'. Allowed: {sorted(VALID_HUMAN_ESCALATION)}"
            )
        if acc not in VALID_HUMAN_ACCURACY:
            errors.append(
                f"Row {row_num}: Invalid accuracy '{acc}'. Allowed: {sorted(VALID_HUMAN_ACCURACY)}"
            )
        if tone not in VALID_HUMAN_TONE:
            errors.append(
                f"Row {row_num}: Invalid tone '{tone}'. Allowed: {sorted(VALID_HUMAN_TONE)}"
            )

    if unfilled:
        print(f"\n⚠️  Found {len(unfilled)} unfilled row(s) in Human QA Benchmark:")
        for r_num, snip, fields in unfilled[:8]:
            print(f"   - Row {r_num} (\"{snip}...\"): Missing {', '.join(fields)}")
        if len(unfilled) > 8:
            print(f"   ... and {len(unfilled) - 8} more.")
        print("\n   👉 Please fill these rows in data/processed/human_qa_benchmark.csv")
        print("      Ensure annotations follow the correct format.")
        return False

    if errors:
        print(f"\n❌ Found {len(errors)} validation error(s):")
        for err in errors[:10]:
            print(f"   {err}")
        return False

    print("\n Human Expected Escalation Distribution:")
    for val, count in df["human_expected_escalation"].value_counts().items():
        print(f"   - {val:<25}: {count:>2} rows")

    print("\n Human Target Accuracy Distribution:")
    for val, count in df["human_target_accuracy"].value_counts().items():
        print(f"   - {val:<25}: {count:>2} rows")

    print("\n Human Target Tone Distribution:")
    for val, count in df["human_target_tone"].value_counts().items():
        print(f"   - {val:<25}: {count:>2} rows")

    print("\n✅ Human QA Benchmark validation passed!")
    return True


def main():
    print("=" * 70)
    print("SUPPORT AGENT ANNOTATION AUDIT & IMPORT TOOL")
    print("=" * 70)

    g_ok = validate_golden_set()
    b_ok = validate_human_benchmark()

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    if g_ok and b_ok:
        print("🎉 ALL CHECKS PASSED! Your datasets are clean and valid.")
        print("\nNext step to run the evaluation harness:")
        print("   python scripts/phase3_eval.py --n-eval 30")
    else:
        print("⚠️  Action required: Please address the issues listed above.")
        print("   Ensure all rows are correctly labeled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
