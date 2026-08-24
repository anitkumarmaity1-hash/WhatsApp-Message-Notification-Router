"""
Lightweight local evaluator.

IMPORTANT: dataset/sample_messages.csv and dataset/messages.csv contain
DIFFERENT messages with unrelated, independently-numbered message_ids
("sample_msg_003" is not the same message as "msg_003" -- they just happen
to share the trailing digit "3"). sample_messages.csv exists only "to
understand the expected output format and style" (see problem_statement.md)
-- there is no ground truth available for messages.csv/output.csv at all;
that comparison happens on hidden labels after submission.

The only thing we CAN honestly self-check locally is: does our own
pipeline, run on sample_messages.csv's own inputs, reproduce the labels
that ship alongside them? That's what this script does -- it re-runs the
real routing pipeline (same code path as main.py) against
dataset/sample_messages.csv and compares each row's prediction to that
same row's own action/message_type, matched by IDENTICAL message_id.

Usage:
    python evaluation/main.py
"""
import csv
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_DIR  # noqa: E402
from data_loader import load_data  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

PRED_OUTPUT_PATH = DATASET_DIR / "sample_predictions.csv"


def _read_rows(path: Path) -> dict:
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[str(row["message_id"])] = row
    return rows


def _write_rows(path: Path, rows: list) -> None:
    cols = ["message_id", "action", "message_type",
            "reason", "confidence", "evidence_message_ids", "model_used"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})


def main():
    sample_path = DATASET_DIR / "sample_messages.csv"
    if not sample_path.exists():
        print("Need dataset/sample_messages.csv to evaluate.")
        return

    print("[eval] loading dataset and swapping in sample_messages.csv "
          "as the routed set ...")
    data_context = load_data()
    if data_context.sample_messages.empty:
        print("sample_messages.csv is empty or failed to load.")
        return

    # Run the REAL pipeline (same code as main.py) but route the labeled
    # sample rows instead of messages.csv, so context lookups (user,
    # group, business, history, evidence) are the genuine ones.
    sample_as_input = data_context.sample_messages.drop(
        columns=["action", "message_type", "reason",
                 "confidence", "evidence_message_ids"],
        errors="ignore",
    )
    eval_context = replace(data_context, messages=sample_as_input)
    results = run_pipeline(eval_context)
    _write_rows(PRED_OUTPUT_PATH, results)
    print(f"[eval] wrote predictions to {PRED_OUTPUT_PATH}")

    truth = _read_rows(sample_path)
    preds = {str(r["message_id"]): r for r in results}

    overlap = [mid for mid in truth if mid in preds]
    if not overlap:
        print("No overlapping message_ids -- pipeline did not return a "
              "prediction for every sample row.")
        return

    action_correct = sum(
        1 for mid in overlap if preds[mid]["action"] == truth[mid]["action"])
    type_correct = sum(
        1 for mid in overlap if preds[mid]["message_type"] == truth[mid]["message_type"])

    n = len(overlap)
    missing = [mid for mid in truth if mid not in preds]

    print(f"Evaluated on {n} labeled sample rows")
    if missing:
        print(
            f"  WARNING: {len(missing)} sample id(s) had no matching prediction: {missing}")
    print(
        f"  action accuracy:       {action_correct}/{n} = {action_correct/n:.2%}")
    print(
        f"  message_type accuracy: {type_correct}/{n} = {type_correct/n:.2%}")

    mismatches = [mid for mid in overlap if preds[mid]
                  ["action"] != truth[mid]["action"]]
    if mismatches:
        print("\nSample mismatches (action):")
        for mid in mismatches[:10]:
            print(
                f"  id={mid} predicted={preds[mid]['action']} expected={truth[mid]['action']}")

    type_mismatches = [mid for mid in overlap if preds[mid]
                       ["message_type"] != truth[mid]["message_type"]]
    if type_mismatches:
        print("\nSample mismatches (message_type):")
        for mid in type_mismatches[:10]:
            print(
                f"  id={mid} predicted={preds[mid]['message_type']} expected={truth[mid]['message_type']}")

    # Accuracy broken down by which tier actually produced the decision.
    by_model: dict = {}
    for mid in overlap:
        m = preds[mid].get("model_used", "unknown")
        by_model.setdefault(m, {"n": 0, "action_ok": 0, "type_ok": 0})
        by_model[m]["n"] += 1
        if preds[mid]["action"] == truth[mid]["action"]:
            by_model[m]["action_ok"] += 1
        if preds[mid]["message_type"] == truth[mid]["message_type"]:
            by_model[m]["type_ok"] += 1

    if len(by_model) > 1:
        print("\nAccuracy by model tier:")
        for model, stats in sorted(by_model.items(), key=lambda kv: -kv[1]["n"]):
            n = stats["n"]
            print(
                f"  {model}: n={n}  action={stats['action_ok']}/{n}="
                f"{stats['action_ok']/n:.0%}  message_type={stats['type_ok']}/{n}="
                f"{stats['type_ok']/n:.0%}"
            )


if __name__ == "__main__":
    main()
