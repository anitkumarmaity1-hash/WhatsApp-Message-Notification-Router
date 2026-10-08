"""
Local evaluator: runs the REAL pipeline on the labeled rows of dataset/sample_messages.csv
and compares each prediction with that row's own action/message_type.

(sample_messages.csv and messages.csv have unrelated IDs; there are no local labels for
messages.csv, so this 30-row check is the only honest local signal available.)

What changed:
  * A run in which the LLM failed is DETECTED and flagged INVALID (the old script happily
    reported ~53% "accuracy" for an all-"digest / 0.2" fallback run). Invalid runs are written
    to a separate file and never overwrite sample_predictions.csv; exit code is 1.
  * --no-llm evaluates only the deterministic layer (rules + fallback) with zero API tokens,
    so you can regression-test feature/rule changes for free.
  * 95% Wilson intervals show how noisy a 30-row score is; per-class recall, confusion counts
    and a confidence-calibration check are added.

Usage:
    python evaluation/main.py            # real run (needs LLM access)
    python evaluation/main.py --no-llm   # deterministic layer only, no tokens used
"""
import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET_DIR  # noqa: E402
from data_loader import load_data  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

LABEL_COLS = ["action", "message_type", "reason",
              "confidence", "evidence_message_ids"]
OUT_COLS = ["message_id", "action", "message_type", "reason", "confidence",
            "evidence_message_ids", "model_used"]
NON_LLM = {"pre_llm_heuristic", "rule_based_fallback", "none"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%}  (95% CI {lo:.0%}-{hi:.0%})"


def write_rows(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUT_COLS})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="evaluate rules/fallback only (0 tokens)")
    args = ap.parse_args()

    if args.no_llm:
        import router.decision_engine as de

        def _no_llm(*_a, **_k):
            raise RuntimeError("LLM disabled by --no-llm")
        de.call_llm_json = _no_llm

    data = load_data()
    if data.sample_messages.empty:
        print("Need dataset/sample_messages.csv to evaluate.")
        return 1

    truth_df = data.sample_messages
    inputs = truth_df.drop(columns=LABEL_COLS, errors="ignore")
    results = run_pipeline(replace(data, messages=inputs))
    truth = {str(r["message_id"]): r for r in truth_df.to_dict("records")}
    preds = {str(r["message_id"]): r for r in results}
    ids = [m for m in truth if m in preds]
    n = len(ids)
    if not n:
        print("No overlapping message_ids - pipeline returned nothing for the sample rows.")
        return 1

    # ---- is this run a valid evaluation of the LLM system? ----
    tiers = Counter(preds[m].get("model_used", "unknown") for m in ids)
    llm_rows = sum(c for t, c in tiers.items() if t not in NON_LLM)
    fallback_rows = tiers.get("rule_based_fallback", 0) + tiers.get("none", 0)
    needs_llm = n - tiers.get("pre_llm_heuristic", 0)
    valid = args.no_llm or (needs_llm == 0 or llm_rows / needs_llm >= 0.8)

    a_ok = sum(preds[m]["action"] == truth[m]["action"] for m in ids)
    t_ok = sum(preds[m]["message_type"] == truth[m]
               ["message_type"] for m in ids)

    mode = "RULES-ONLY (--no-llm)" if args.no_llm else "FULL PIPELINE"
    print(f"\n=== {mode} on {n} labeled sample rows ===")
    print("answered by:", dict(tiers))
    print(f"action accuracy:       {fmt(a_ok, n)}")
    print(f"message_type accuracy: {fmt(t_ok, n)}")

    print("\nAction recall per class (true -> predicted counts):")
    for cls in ("notify", "digest", "mute"):
        rows = [m for m in ids if truth[m]["action"] == cls]
        if rows:
            dist = Counter(preds[m]["action"] for m in rows)
            print(
                f"  {cls:7} recall {sum(preds[m]['action'] == cls for m in rows)}/{len(rows)}   -> {dict(dist)}")

    ok = [float(preds[m]["confidence"])
          for m in ids if preds[m]["action"] == truth[m]["action"]]
    bad = [float(preds[m]["confidence"])
           for m in ids if preds[m]["action"] != truth[m]["action"]]
    if ok and bad:
        print(f"\nCalibration: mean confidence when right {sum(ok) / len(ok):.2f} vs wrong "
              f"{sum(bad) / len(bad):.2f} (a good system is clearly lower when wrong)")

    print("\nBy tier:")
    for tier, c in tiers.most_common():
        sub = [m for m in ids if preds[m].get("model_used", "unknown") == tier]
        print(f"  {tier}: n={c} action={sum(preds[m]['action'] == truth[m]['action'] for m in sub)}/{c} "
              f"type={sum(preds[m]['message_type'] == truth[m]['message_type'] for m in sub)}/{c}")

    wrong = [m for m in ids if preds[m]["action"] != truth[m]["action"]]
    if wrong:
        print("\nAction mismatches (first 10):")
        for m in wrong[:10]:
            print(f"  {m}: predicted {preds[m]['action']}/{preds[m]['message_type']}  "
                  f"expected {truth[m]['action']}/{truth[m]['message_type']}")

    # ---- output file: never let an invalid (fallback) run masquerade as a real one ----
    if valid:
        path = DATASET_DIR / \
            ("sample_predictions_rules_only.csv" if args.no_llm else "sample_predictions.csv")
    else:
        path = DATASET_DIR / "sample_predictions_INVALID_llm_failed.csv"
        print(f"\n!!! INVALID EVALUATION: only {llm_rows}/{needs_llm} LLM-eligible rows were answered by an LLM "
              f"({fallback_rows} used the blind fallback). The scores above do NOT measure the system.\n"
              f"    Check GROQ_API_KEY / daily token budget / Ollama, then re-run. "
              f"(Use --no-llm to evaluate the deterministic layer on purpose.)")
    write_rows(path, results)
    print(f"\n[eval] wrote predictions to {path}")
    print(
        f"[eval] note: n={n} is small; treat differences under ~15 points as noise.")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
