"""Merge O*NET task frame with AEI usage and OEWS employment data."""

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent

# --- Config ---
RELEASE = "release_2026_03_24"
ONET_CSV = BASE / "data" / "onet" / "task_statements.csv"
AEI_CSV = BASE / "data" / RELEASE / "aei_cleaned_claude_ai.csv"
GROUP_2010_CSV = BASE / "data" / "onet" / "soc_2010_to_group.csv"
GROUP_2018_CSV = BASE / "data" / "onet" / "soc_2018_to_group.csv"
OEWS_CSV = BASE / "data" / "oews" / "national_M2024_dl.csv"
OUT_CSV = BASE / "data" / "merged.csv"

# ==========================================================================
# Step 1: O*NET ← AEI (left join on task text)
# ==========================================================================
onet = pd.read_csv(ONET_CSV)
aei = pd.read_csv(AEI_CSV)

onet["task_merge_key"] = onet["Task"].str.lower().str.strip()
aei["task_merge_key"] = aei["task"].str.lower().str.strip()

aei_keys = set(aei["task_merge_key"])
onet_keys = set(onet["task_merge_key"])

logger.info("--- Step 1: O*NET ← AEI ---")
logger.info("  O*NET tasks (unique text): %d", len(onet_keys))
logger.info("  AEI tasks (unique text): %d", len(aei_keys))
logger.info("  Matched: %d", len(onet_keys & aei_keys))

aei_only = aei_keys - onet_keys
if aei_only:
    logger.info("  AEI tasks not in O*NET (dropped): %d", len(aei_only))
    for t in sorted(aei_only):
        logger.info("    - %r", t)

onet_only = onet_keys - aei_keys
logger.info("  O*NET tasks not in AEI (zero-filled): %d", len(onet_only))

aei = aei.drop(columns=["task"])
tasks = onet.merge(aei, on="task_merge_key", how="left")
tasks = tasks.drop(columns=["task_merge_key"])

# Fill zeros for count columns and task_pct (collaboration pcts stay NaN)
fill_cols = [c for c in tasks.columns if c.endswith("_count")]
fill_cols.append("task_pct")
tasks[fill_cols] = tasks[fill_cols].fillna(0)

n_with_aei = tasks["task_count"].gt(0).sum()
n_without_aei = tasks["task_count"].eq(0).sum()
logger.info(
    "  Result: %d rows (%d with AEI data, %d zero-filled)",
    len(tasks),
    n_with_aei,
    n_without_aei,
)

# ==========================================================================
# Step 2: Add group_id via SOC 2010 crosswalk
# ==========================================================================
g2010 = pd.read_csv(GROUP_2010_CSV)

tasks["soc_2010"] = tasks["O*NET-SOC Code"].str[:7]
tasks = tasks.merge(g2010[["soc_2010", "group_id"]], on="soc_2010", how="left")

n_grouped = tasks["group_id"].notna().sum()
n_ungrouped = tasks["group_id"].isna().sum()
logger.info("--- Step 2: SOC 2010 → group_id ---")
logger.info("  Tasks grouped: %d, ungrouped: %d", n_grouped, n_ungrouped)

# For ungrouped tasks, try prefix match (broad SOC codes like 19-1020).
# Store broad-code child groups for coarsening in step 3b.
task_broad_codes = {}  # soc_2010 -> list of child group_ids
if n_ungrouped > 0:
    ungrouped_socs = tasks.loc[tasks["group_id"].isna(), "soc_2010"].unique()
    for s in sorted(ungrouped_socs):
        title = tasks.loc[tasks["soc_2010"] == s, "Title"].iloc[0]
        prefix = s[:6]
        children = g2010[g2010["soc_2010"].str.startswith(prefix)]
        if not children.empty:
            child_groups = sorted(int(g) for g in children["group_id"].unique())
            # Temporarily assign first child group; step 3b will coarsen if needed
            tasks.loc[tasks["soc_2010"] == s, "group_id"] = child_groups[0]
            task_broad_codes[s] = child_groups
            logger.info(
                "    - %s (%s) -> prefix %s*, child groups: %s",
                s,
                title,
                prefix,
                child_groups,
            )
        else:
            logger.info("    - %s (%s) -> no match", s, title)

task_group_ids = set(tasks.loc[tasks["group_id"].notna(), "group_id"].astype(int))
logger.info("  Unique groups represented: %d", len(task_group_ids))

# ==========================================================================
# Step 3: OEWS ← group_id via SOC 2018 crosswalk
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
oews = oews.merge(
    g2018[["soc_2018", "group_id"]], left_on="occ_code", right_on="soc_2018", how="left"
)

oews_group_ids = set(oews.loc[oews["group_id"].notna(), "group_id"].astype(int))
n_oews_grouped = oews["group_id"].notna().sum()
n_oews_ungrouped = oews["group_id"].isna().sum()
logger.info("--- Step 3: OEWS ← group_id ---")
logger.info("  OEWS detailed occupations: %d", len(oews))
logger.info("  Grouped: %d, ungrouped: %d", n_oews_grouped, n_oews_ungrouped)
logger.info("  Unique groups represented: %d", len(oews_group_ids))
if n_oews_ungrouped > 0:
    ungrouped_emp = oews.loc[oews["group_id"].isna(), "tot_emp"].sum()
    logger.info("  Ungrouped employment: %s", f"{ungrouped_emp:,.0f}")
    logger.info("  OEWS codes not in crosswalk:")
    for _, row in oews[oews["group_id"].isna()].iterrows():
        logger.info(
            "    - %s %s (emp: %s)",
            row["occ_code"],
            row["occ_title"],
            f"{row['tot_emp']:,.0f}",
        )

# --- Step 3b: Coarsened crosswalk for broad codes on either side ---
# When OEWS or O*NET uses a broad code (e.g., 31-1120 or 19-1020) instead of
# detailed codes, we merge the child groups into a new coarsened group.
# Both sides get coarsened symmetrically.
tasks["group_id_coarsened"] = tasks["group_id"].copy()
oews["group_id_coarsened"] = oews["group_id"].copy()

logger.info("--- Step 3b: Coarsened crosswalk for broad codes ---")
next_group_id = int(max(g2010["group_id"].max(), g2018["group_id"].max())) + 1
coarsened_count = 0

# Collect all coarsenings needed: from OEWS ungrouped codes and task broad codes
coarsenings = []  # list of dicts with child_groups and metadata

# From OEWS side: ungrouped broad codes
for _, row in oews[oews["group_id_coarsened"].isna()].iterrows():
    code = row["occ_code"]
    prefix = code[:6]
    children = g2018[g2018["soc_2018"].str.startswith(prefix)]
    if children.empty:
        logger.info("  %s: no children found with prefix %s, skipping", code, prefix)
        continue
    child_groups = set(int(g) for g in children["group_id"].unique())
    child_codes = ", ".join(children["soc_2018"].tolist())
    coarsenings.append(
        {
            "label": f"OEWS {code}",
            "child_groups": child_groups,
            "oews_code": code,
            "detail": child_codes,
        }
    )

# From task side: broad SOC 2010 codes with multiple child groups
for soc_2010, child_groups in task_broad_codes.items():
    if len(child_groups) > 1:
        coarsenings.append(
            {
                "label": f"O*NET {soc_2010}",
                "child_groups": set(child_groups),
                "oews_code": None,
                "detail": soc_2010,
            }
        )

# Merge overlapping coarsenings via connected components.
# Two coarsenings that share a child group must become a single coarsened group.
G_coarsen = nx.Graph()
for i, c in enumerate(coarsenings):
    for g in c["child_groups"]:
        G_coarsen.add_edge(f"c{i}", f"g{g}")

merged_coarsenings = []
for component in nx.connected_components(G_coarsen):
    # Gather all coarsening entries and child groups in this component
    entry_indices = [int(n[1:]) for n in component if n.startswith("c")]
    all_child_groups = set()
    labels = []
    oews_codes = []
    details = []
    for i in sorted(entry_indices):
        all_child_groups |= coarsenings[i]["child_groups"]
        labels.append(coarsenings[i]["label"])
        if coarsenings[i]["oews_code"]:
            oews_codes.append(coarsenings[i]["oews_code"])
        details.append(coarsenings[i]["detail"])
    merged_coarsenings.append(
        {
            "labels": labels,
            "child_groups": sorted(all_child_groups),
            "oews_codes": oews_codes,
            "detail": "; ".join(details),
        }
    )

# Apply merged coarsenings
for mc in merged_coarsenings:
    child_groups = mc["child_groups"]
    new_gid = next_group_id
    next_group_id += 1

    # Remap tasks in child groups to new coarsened group
    tasks.loc[tasks["group_id"].isin(child_groups), "group_id_coarsened"] = new_gid

    # Remap OEWS: broad codes and any detailed codes in child groups
    for oews_code in mc["oews_codes"]:
        oews.loc[oews["occ_code"] == oews_code, "group_id_coarsened"] = new_gid
    oews.loc[oews["group_id"].isin(child_groups), "group_id_coarsened"] = new_gid

    label_str = " + ".join(mc["labels"])
    logger.info(
        "  %s -> group %d (merged %s; %s)",
        label_str,
        new_gid,
        child_groups,
        mc["detail"],
    )
    coarsened_count += 1

logger.info("  Coarsened %d OEWS codes", coarsened_count)

# Recompute group overlap after coarsening
task_coarsened_ids = set(
    tasks.loc[tasks["group_id_coarsened"].notna(), "group_id_coarsened"].astype(int)
)
oews_coarsened_ids = set(
    oews.loc[oews["group_id_coarsened"].notna(), "group_id_coarsened"].astype(int)
)
shared_groups = task_coarsened_ids & oews_coarsened_ids
tasks_only_groups = task_coarsened_ids - oews_coarsened_ids
oews_only_groups = oews_coarsened_ids - task_coarsened_ids
logger.info("  Group overlap (after coarsening):")
logger.info("    Shared: %d", len(shared_groups))
logger.info("    Task-side only (no OEWS employment): %d", len(tasks_only_groups))
logger.info("    OEWS-side only (no O*NET tasks): %d", len(oews_only_groups))

# ==========================================================================
# Step 4: Aggregate OEWS by group_id_coarsened
# ==========================================================================
oews_with_group = oews.dropna(subset=["group_id_coarsened"]).copy()
oews_with_group["group_id_coarsened"] = oews_with_group["group_id_coarsened"].astype(
    int
)


# Employment-weighted mean wage (skip NaN wages, return NaN if all wages missing)
def emp_weighted_mean(df):
    mask = df["a_mean"].notna()
    if not mask.any():
        return np.nan
    return np.average(df.loc[mask, "a_mean"], weights=df.loc[mask, "tot_emp"])


oews_by_group = (
    oews_with_group.groupby("group_id_coarsened")
    .agg(
        coarsened_group_tot_emp=("tot_emp", "sum"),
        coarsened_group_n_occ=("occ_code", "nunique"),
    )
    .reset_index()
)
oews_by_group["coarsened_group_a_mean"] = (
    oews_with_group.groupby("group_id_coarsened").apply(emp_weighted_mean).values
)

logger.info("--- Step 4: OEWS aggregation by coarsened group ---")
logger.info("  Groups with employment: %d", len(oews_by_group))
logger.info(
    "  Total employment: %s", f"{oews_by_group['coarsened_group_tot_emp'].sum():,.0f}"
)
n_with_wage = oews_by_group["coarsened_group_a_mean"].notna().sum()
n_without_wage = oews_by_group["coarsened_group_a_mean"].isna().sum()
logger.info("  Groups with wage data: %d", n_with_wage)
logger.info("  Groups without wage data: %d", n_without_wage)

# ==========================================================================
# Step 5: Merge OEWS groups onto task frame
# ==========================================================================
tasks = tasks.merge(oews_by_group, on="group_id_coarsened", how="left")

n_with_emp = tasks["coarsened_group_tot_emp"].notna().sum()
n_without_emp = tasks["coarsened_group_tot_emp"].isna().sum()
emp_covered = (
    tasks.loc[tasks["coarsened_group_tot_emp"].notna(), "coarsened_group_tot_emp"]
    .drop_duplicates()
    .sum()
)
logger.info("--- Step 5: Tasks ← OEWS groups ---")
logger.info("  Tasks with employment data: %d", n_with_emp)
logger.info("  Tasks without employment data: %d", n_without_emp)
logger.info("  Total employment covered: %s", f"{emp_covered:,.0f}")
if n_without_emp > 0:
    no_emp = tasks.loc[tasks["coarsened_group_tot_emp"].isna()]
    n_with_aei_no_emp = no_emp["task_count"].gt(0).sum()
    n_occ_no_emp = no_emp["O*NET-SOC Code"].nunique()
    logger.info(
        "  Of which have AEI data but no employment: %d tasks across %d occupations",
        n_with_aei_no_emp,
        n_occ_no_emp,
    )

# ==========================================================================
# Step 6: Apportion employment to occupations
# ==========================================================================
# Count unique SOC 2010 codes per coarsened group, then divide equally.
n_occ_per_group = (
    tasks[tasks["group_id_coarsened"].notna()]
    .groupby("group_id_coarsened")["soc_2010"]
    .nunique()
    .rename("n_soc_2010_in_group")
)
tasks = tasks.merge(n_occ_per_group, on="group_id_coarsened", how="left")
tasks["apportioned_occ_emp"] = (
    tasks["coarsened_group_tot_emp"] / tasks["n_soc_2010_in_group"]
)

logger.info("--- Step 6: Apportion employment to occupations ---")
n_occs_with_emp = tasks.loc[tasks["apportioned_occ_emp"].notna(), "soc_2010"].nunique()
total_apportioned = (
    tasks[tasks["apportioned_occ_emp"].notna()]
    .drop_duplicates("soc_2010")["apportioned_occ_emp"]
    .sum()
)
logger.info("  SOC 2010 occupations with employment: %d", n_occs_with_emp)
logger.info("  Total apportioned employment: %s", f"{total_apportioned:,.0f}")

# ==========================================================================
# Step 7: Apportion usage counts across task-occupations
# ==========================================================================
# task_count is per unique task text. When a task appears under multiple
# occupations, split its count proportional to occupation employment.
# For tasks with no employment data, count stays undivided.

tasks["_task_key"] = tasks["Task"].str.lower().str.strip()

# Total employment across all occupations sharing each task text
task_total_emp = (
    tasks[tasks["apportioned_occ_emp"].notna()]
    .groupby("_task_key")["apportioned_occ_emp"]
    .transform("sum")
)

tasks["apportioned_task_count"] = tasks["task_count"]  # default: undivided
mask = tasks["apportioned_occ_emp"].notna() & (task_total_emp > 0)
tasks.loc[mask, "apportioned_task_count"] = (
    tasks.loc[mask, "task_count"]
    * tasks.loc[mask, "apportioned_occ_emp"]
    / task_total_emp[mask]
)
tasks = tasks.drop(columns=["_task_key"])

logger.info("--- Step 7: Apportion usage counts across task-occupations ---")
orig_sum = tasks.drop_duplicates("Task")["task_count"].sum()
apportioned_sum = tasks["apportioned_task_count"].sum()
logger.info("  Original task_count sum (unique tasks): %s", f"{orig_sum:,.0f}")
logger.info("  Apportioned task_count sum (all rows): %s", f"{apportioned_sum:,.0f}")
n_split = (tasks["apportioned_task_count"] != tasks["task_count"]).sum()
logger.info("  Task-occupation pairs with split count: %d", n_split)

# ==========================================================================
# Write
# ==========================================================================
tasks.to_csv(OUT_CSV, index=False)
logger.info("Wrote %d rows -> %s", len(tasks), OUT_CSV)
