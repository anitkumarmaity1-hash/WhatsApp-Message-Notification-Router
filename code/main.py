"""
Entry point for the Message Notification Router.

Usage:
    python main.py
"""
from collections import Counter

from data_loader import load_data
from pipeline import run_pipeline
from output_writer import write_output


def main():
    print("[main] loading dataset ...")
    data_context = load_data()
    print(f"[main] {len(data_context.messages)} messages to route")

    results = run_pipeline(data_context)
    write_output(results)

    action_counts = Counter(r["action"] for r in results)
    type_counts = Counter(r["message_type"] for r in results)
    print("\n[main] action distribution:", dict(action_counts))
    print("[main] message_type distribution:", dict(type_counts))


if __name__ == "__main__":
    main()
