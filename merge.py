"""Merge O*NET task frame with AEI usage data, filling zeros for unmatched tasks."""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent

# --- Config ---
RELEASE = "release_2026_03_24"
ONET_CSV = BASE / "data" / "onet" / "task_statements.csv"
AEI_CSV = BASE / "data" / RELEASE / "aei_cleaned_claude_ai.csv"
OUT_CSV = BASE / "data" / "merged.csv"

# --- Load ---
onet = pd.read_csv(ONET_CSV)
aei = pd.read_csv(AEI_CSV)

# --- Prepare merge key (lowercase task text) ---
onet["task_merge_key"] = onet["Task"].str.lower().str.strip()
aei["task_merge_key"] = aei["task"].str.lower().str.strip()

# --- Left merge: O*NET ← AEI ---
aei_keys = set(aei["task_merge_key"])
onet_keys = set(onet["task_merge_key"])

print("--- O*NET ← AEI merge ---")
print(f"  O*NET tasks (unique text): {len(onet_keys)}")
print(f"  AEI tasks (unique text): {len(aei_keys)}")
print(f"  Matched: {len(onet_keys & aei_keys)}")

aei_only = aei_keys - onet_keys
if aei_only:
    print(f"  AEI tasks not in O*NET (dropped): {len(aei_only)}")
    for t in sorted(aei_only):
        print(f"    - {t!r}")

onet_only = onet_keys - aei_keys
print(f"  O*NET tasks not in AEI (zero-filled): {len(onet_only)}")

# Drop the AEI 'task' column (we keep O*NET's 'Task' as canonical)
aei = aei.drop(columns=["task"])

merged = onet.merge(aei, on="task_merge_key", how="left")
merged = merged.drop(columns=["task_merge_key"])

# --- Fill zeros for count columns and task_pct (collaboration pcts stay NaN) ---
fill_cols = [c for c in merged.columns if c.endswith("_count")]
fill_cols.append("task_pct")
merged[fill_cols] = merged[fill_cols].fillna(0)

# --- Report ---
n_matched = merged["task_count"].gt(0).sum()
n_zero = merged["task_count"].eq(0).sum()
print(f"\nMerged: {len(merged)} rows ({n_matched} with AEI data, {n_zero} zero-filled)")

merged.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(merged)} rows -> {OUT_CSV}")
