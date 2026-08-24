import pandas as pd

out = pd.read_csv(
    "C:/Users/anitk/OneDrive/Desktop/HackerRank/hackerrank-orchestrate-august26/dataset/output.csv")
samp = pd.read_csv(
    "C:/Users/anitk/OneDrive/Desktop/HackerRank/hackerrank-orchestrate-august26/dataset/sample_messages.csv")

# extract numeric suffix from each
out_ids = out["message_id"].str.extract(
    r'(\d+)$')[0].astype(int).sort_values().tolist()
samp_ids = samp["message_id"].str.extract(
    r'(\d+)$')[0].astype(int).sort_values().tolist()

print("output numeric range:", min(out_ids),
      "-", max(out_ids), "count:", len(out_ids))
print("sample numeric range:", min(samp_ids),
      "-", max(samp_ids), "count:", len(samp_ids))
print("output ids:", out_ids)
print("sample ids:", samp_ids)
