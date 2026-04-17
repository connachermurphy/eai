"""Clean the AEI Claude.ai data: filter, pivot wide, and merge onto O*NET tasks."""

from pathlib import Path

import pandas as pd

# --- Config ---
RELEASE = "release_2026_03_24"
BASE = Path(__file__).resolve().parent.parent
RAW_CSV = BASE / "data" / RELEASE / "aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv"
ONET_CSV = BASE / "data" / "onet" / "task_statements.csv"
OUT_CSV = BASE / "data" / RELEASE / "aei_cleaned_claude_ai.csv"

# --- Load & Filter ---
raw = pd.read_csv(RAW_CSV)
# TODO: add country-level geo_ids once we decide how to handle aggregation
raw = raw[raw["geo_id"] == "GLOBAL"]

df = raw[raw["facet"] == "onet_task::collaboration"].copy()

# --- Task usage totals (from onet_task facet) ---
usage = raw[raw["facet"] == "onet_task"].copy()
usage["var_short"] = usage["variable"].str.replace("onet_task_", "")
usage_wide = (
    usage.pivot_table(index="cluster_name", columns="var_short", values="value", aggfunc="first")
    .rename(columns={"count": "task_count", "pct": "task_pct"})
    .reset_index()
    .rename(columns={"cluster_name": "task"})
)

# --- Parse cluster_name ---
df["task"] = df["cluster_name"].str.rsplit("::", n=1).str[0]
df["collaboration_type"] = df["cluster_name"].str.rsplit("::", n=1).str[1]
df["collaboration_type"] = df["collaboration_type"].str.replace(" ", "_")

# Shorten variable names: "onet_task_collaboration_count" -> "count"
df["var_short"] = df["variable"].str.replace("onet_task_collaboration_", "")
df["col_name"] = df["collaboration_type"] + "_" + df["var_short"]

# --- Pivot Wide ---
dupes = df.duplicated(subset=["task", "col_name"], keep=False)
assert not dupes.any(), f"Duplicate (task, col_name) pairs:\n{df.loc[dupes, ['task', 'col_name']]}"

wide = (
    df.pivot_table(index="task", columns="col_name", values="value", aggfunc="first")
    .fillna(0)
    .reset_index()
)

# --- Merge O*NET ---
onet = pd.read_csv(ONET_CSV)
onet["task_merge_key"] = onet["Task"].str.lower().str.strip()
onet = onet[["task_merge_key", "O*NET-SOC Code", "Title", "Task ID"]].drop_duplicates()
onet = onet.rename(
    columns={
        "O*NET-SOC Code": "onet_soc_code",
        "Title": "onet_title",
        "Task ID": "onet_task_id",
    }
)

wide["task_merge_key"] = wide["task"].str.lower().str.strip()
onet_keys = set(onet["task_merge_key"])
wide_keys = set(wide["task_merge_key"])
onet_unmatched = onet_keys - wide_keys
print(f"\n--- O*NET merge ---")
print(f"  O*NET tasks: {len(onet_keys)}")
print(f"  AEI tasks: {len(wide_keys)}")
print(f"  O*NET tasks not in AEI (dropped from right): {len(onet_unmatched)}")
aei_unmatched_onet = wide_keys - onet_keys
# TODO: handle tasks with no O*NET match (currently just the literal "none" task)
if aei_unmatched_onet:
    print(f"  AEI tasks not in O*NET (null onet columns): {len(aei_unmatched_onet)}")
    for t in sorted(aei_unmatched_onet):
        print(f"    - {t!r}")

wide = wide.merge(onet, on="task_merge_key", how="left")

# --- Merge task usage totals ---
usage_wide["task_merge_key"] = usage_wide["task"].str.lower().str.strip()
usage_keys = set(usage_wide["task_merge_key"])
usage_unmatched = usage_keys - wide_keys
print(f"\n--- Task usage merge ---")
print(f"  Usage tasks: {len(usage_keys)}")
print(f"  AEI tasks: {len(wide_keys)}")
print(f"  Usage tasks not in AEI (dropped from right): {len(usage_unmatched)}")
if usage_unmatched:
    for t in sorted(usage_unmatched):
        print(f"    - {t!r}")
aei_unmatched_usage = wide_keys - usage_keys
if aei_unmatched_usage:
    print(f"  AEI tasks not in usage (null task_count/task_pct): {len(aei_unmatched_usage)}")
    for t in sorted(aei_unmatched_usage):
        print(f"    - {t!r}")

wide = wide.merge(
    usage_wide[["task_merge_key", "task_count", "task_pct"]],
    on="task_merge_key",
    how="left",
)
wide = wide.drop(columns=["task_merge_key"])
print()

# --- Order columns: task, counts, pcts, onet ---
collab_types = ["directive", "feedback_loop", "learning", "none", "not_classified", "task_iteration", "validation"]
count_cols = [f"{t}_count" for t in collab_types]
pct_cols = [f"{t}_pct" for t in collab_types]
wide = wide[["task", "task_count", "task_pct"] + count_cols + pct_cols + ["onet_soc_code", "onet_title", "onet_task_id"]]

# --- Write ---
wide.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(wide)} rows to {OUT_CSV}")
