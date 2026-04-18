"""Merge O*NET task frame with AEI usage and OEWS employment data."""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent

# --- Config ---
RELEASE = "release_2026_03_24"
ONET_CSV = BASE / "data" / "onet" / "task_statements.csv"
AEI_CSV = BASE / "data" / RELEASE / "aei_cleaned_claude_ai.csv"
GROUP_2010_CSV = BASE / "data" / "onet" / "soc_2010_to_group.csv"
GROUP_2018_CSV = BASE / "data" / "onet" / "soc_2018_to_group.csv"
OEWS_CSV = BASE / "data" / "oews" / "national_M2024_dl.csv"
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

# ==========================================================================
# Step 2: Add group_id via SOC 2010 crosswalk
# ==========================================================================
g2010 = pd.read_csv(GROUP_2010_CSV)

# O*NET-SOC "11-1011.00" -> SOC 2010 "11-1011"
merged["soc_2010"] = merged["O*NET-SOC Code"].str[:7]

merged = merged.merge(g2010[["soc_2010", "group_id"]], on="soc_2010", how="left")

n_grouped = merged["group_id"].notna().sum()
n_ungrouped = merged["group_id"].isna().sum()
print(f"\n--- Group ID assignment ---")
print(f"  Grouped: {n_grouped}, ungrouped: {n_ungrouped}")
if n_ungrouped > 0:
    ungrouped_socs = merged.loc[merged["group_id"].isna(), "soc_2010"].unique()
    for s in sorted(ungrouped_socs):
        print(f"    - {s}")

# ==========================================================================
# Step 3: Merge OEWS employment via group_id
# ==========================================================================
g2018 = pd.read_csv(GROUP_2018_CSV)
oews = pd.read_csv(OEWS_CSV)

# Filter to detailed occupations only
oews = oews[oews["o_group"] == "detailed"].copy()
oews = oews[["occ_code", "occ_title", "tot_emp", "a_mean"]].copy()

# Clean numeric columns (* = not available, leave as NaN)
oews["tot_emp"] = pd.to_numeric(oews["tot_emp"], errors="coerce")
oews["a_mean"] = pd.to_numeric(oews["a_mean"], errors="coerce")

# Join group_id onto OEWS
oews = oews.merge(g2018[["soc_2018", "group_id"]], left_on="occ_code", right_on="soc_2018", how="left")

n_oews_grouped = oews["group_id"].notna().sum()
n_oews_ungrouped = oews["group_id"].isna().sum()
print(f"\n--- OEWS ← group_id ---")
print(f"  Grouped: {n_oews_grouped}, ungrouped: {n_oews_ungrouped}")
if n_oews_ungrouped > 0:
    print("  Ungrouped OEWS codes (no SOC 2010 equivalent):")
    for _, row in oews[oews["group_id"].isna()].iterrows():
        print(f"    - {row['occ_code']} {row['occ_title']}")

# ==========================================================================
# Step 4: Aggregate OEWS by group_id (sum employment, employment-weighted mean wage)
# ==========================================================================
oews_grouped = oews.dropna(subset=["group_id"]).copy()
oews_grouped["group_id"] = oews_grouped["group_id"].astype(int)

# Employment-weighted mean wage
oews_grouped["aggregate_comp"] = oews_grouped["a_mean"] * oews_grouped["tot_emp"]

oews_by_group = (
    oews_grouped.groupby("group_id")
    .agg(
        group_tot_emp=("tot_emp", "sum"),
        group_n_occ=("occ_code", "nunique"),
        _aggregate_comp=("aggregate_comp", "sum"),
    )
    .reset_index()
)
oews_by_group["group_a_mean"] = (
    oews_by_group["_aggregate_comp"] / oews_by_group["group_tot_emp"]
)
oews_by_group = oews_by_group.drop(columns=["_aggregate_comp"])

print(f"\n--- OEWS aggregation by group ---")
print(f"  Groups with employment: {len(oews_by_group)}")
print(f"  Total employment: {oews_by_group['group_tot_emp'].sum():,.0f}")

# TODO: Merge onto task frame and apportion employment across O*NET occupations

# ==========================================================================
# Write
# ==========================================================================
merged.to_csv(OUT_CSV, index=False)
print(f"\nWrote {len(merged)} rows -> {OUT_CSV}")
