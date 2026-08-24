"""Writes the final predictions to dataset/output.csv with the exact
required column order."""
import csv
from config import OUTPUT_PATH

REQUIRED_COLUMNS = ["message_id", "action", "message_type",
                    "reason", "confidence", "evidence_message_ids"]


def write_output(rows: list) -> None:
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REQUIRED_COLUMNS})
    print(f"[output_writer] wrote {len(rows)} rows to {OUTPUT_PATH}")
