"""Create merged occupational characteristics datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

from eai.aei import load_aei_tasks
from eai.utils import get_logger, log_merge_diagnostics, merge_with_diagnostics

log = get_logger(__name__)
DATA = Path("output")
OUT = Path("output")

# ======================================================================================
# Step 1: Load both SOC universes from the connected-component crosswalk
# ======================================================================================
soc_2018 = pd.read_csv(DATA / "onet" / "soc_2018_to_group.csv")
broad_codes = soc_2018[soc_2018["soc_2018"].str[-1] == "0"]
assert broad_codes.empty, (
    f"Universe contains {len(broad_codes)} broad codes (ending in 0): "
    f"{broad_codes['soc_2018'].tolist()}"
)
log.info(
    "SOC 2018 universe: %d occupations, %d groups",
    len(soc_2018),
    soc_2018["group_id"].nunique(),
)

soc_2010 = pd.read_csv(DATA / "onet" / "soc_2010_to_group.csv")
xwalk_edges = pd.read_csv(DATA / "onet" / "soc_crosswalk_edges.csv")
broad_codes_2010 = soc_2010[soc_2010["soc_2010"].str[-1] == "0"]
assert broad_codes_2010.empty, (
    f"Universe contains {len(broad_codes_2010)} broad codes (ending in 0): "
    f"{broad_codes_2010['soc_2010'].tolist()}"
)
log.info(
    "SOC 2010 universe: %d occupations, %d groups",
    len(soc_2010),
    soc_2010["group_id"].nunique(),
)
log.info("Crosswalk edges: %d direct SOC 2010<->2018 mappings", len(xwalk_edges))

# O*NET task statements: shared dependency for SOC 2010 side.
# Truncate 8-digit O*NET-SOC codes to 6-digit SOC 2010. This implicitly collapses
# sub-occupations (e.g., 15-1199.01, 15-1199.02) into their parent SOC code.
# For task-level AEI data this is fine: counts are additive across sub-occupations
# and get summed when we aggregate to occupation level in step 7.
onet = pd.read_csv(DATA / "onet" / "task_statements.csv")
onet["task_key"] = onet["Task"].str.lower().str.strip()
onet["soc_2010"] = onet["O*NET-SOC Code"].str[:7]
task_occ = onet[["task_key", "soc_2010"]].drop_duplicates()
task_occ_occupations = task_occ["soc_2010"].unique()
occs_per_task = task_occ.groupby("task_key")["soc_2010"].nunique()
n_shared = (occs_per_task > 1).sum()
n_unique = (occs_per_task == 1).sum()
log.info(
    "O*NET task statements: %d unique tasks"
    " (%d shared, %d unique to one), %d occupations",
    task_occ["task_key"].nunique(),
    n_shared,
    n_unique,
    len(task_occ_occupations),
)

# ======================================================================================
# Step 2: Load 2022 OEWS and build lookup keyed on SOC 2018 (exact + broad)
# ======================================================================================
oews_raw = pd.read_csv(DATA / "oews" / "national_M2022_dl.csv")
oews = oews_raw[oews_raw["o_group"] == "detailed"][
    ["occ_code", "occ_title", "tot_emp", "a_mean", "a_median"]
].copy()
oews = oews.rename(columns={"occ_code": "soc_2018"})
for col in ["tot_emp", "a_mean", "a_median"]:
    oews[col] = pd.to_numeric(oews[col], errors="coerce")
log.info("OEWS 2022: %d detailed occupations", len(oews))

# Exact matches
universe_codes = set(soc_2018["soc_2018"])
oews_codes = set(oews["soc_2018"])

exact = oews[oews["soc_2018"].isin(universe_codes)].copy()
exact["broad_match"] = False
exact["soc_2018_broad"] = pd.NA

# Coarsen unmatched universe codes (last digit -> 0) and match against literal OEWS
unmatched_codes = universe_codes - oews_codes
unmatched_oews = oews[oews["soc_2018"].isin(oews_codes - universe_codes)]

broad_map = pd.DataFrame({"soc_2018": sorted(unmatched_codes)})
broad_map["soc_2018_broad"] = broad_map["soc_2018"].str[:6] + "0"

broad_matched, b_left_only, b_right_only = merge_with_diagnostics(
    broad_map,
    unmatched_oews.rename(columns={"soc_2018": "soc_2018_broad"}),
    on="soc_2018_broad",
)
log_merge_diagnostics(
    b_left_only,
    b_right_only,
    left_label="Unmatched universe (coarsened)",
    right_label="Unmatched OEWS",
    labels=soc_2018,
    key_col="soc_2018",
    logger=log,
)

# Adjust employment: divide by number of universe codes sharing each broad code
broad_matched = broad_matched[broad_matched["tot_emp"].notna()].copy()
n_per_broad = broad_matched.groupby("soc_2018_broad")["soc_2018"].nunique()
broad_matched["broad_match"] = True
broad_matched["tot_emp_adjusted"] = broad_matched["tot_emp"] / (
    broad_matched["soc_2018_broad"].map(n_per_broad)
)

# Combine exact + broad into a single OEWS lookup, with group_id
exact["tot_emp_adjusted"] = exact["tot_emp"]
lookup_cols = [
    "soc_2018",
    "occ_title",
    "tot_emp",
    "a_mean",
    "a_median",
    "broad_match",
    "soc_2018_broad",
    "tot_emp_adjusted",
]
oews_lookup = pd.concat([exact[lookup_cols], broad_matched[lookup_cols]])
oews_lookup = oews_lookup.merge(
    soc_2018[["soc_2018", "group_id"]], on="soc_2018", how="left"
)
log.info(
    "OEWS lookup: %d exact + %d broad = %d total",
    len(exact),
    len(broad_matched),
    len(oews_lookup),
)

# Aggregate OEWS to group level for diagnostics and grouped outputs


def _emp_weighted_mean(sub):
    """Employment-weighted mean wage, returning NaN if all wages missing."""
    mask = sub["a_mean"].notna() & sub["tot_emp_adjusted"].notna()
    if not mask.any():
        return np.nan
    return np.average(
        sub.loc[mask, "a_mean"], weights=sub.loc[mask, "tot_emp_adjusted"]
    )


oews_by_group = (
    oews_lookup.groupby("group_id")
    .agg(
        oews_group_tot_emp=("tot_emp_adjusted", "sum"),
        oews_group_n_soc_2018=("soc_2018", "nunique"),
    )
    .reset_index()
)
oews_by_group["oews_group_a_mean"] = (
    oews_lookup.groupby("group_id").apply(_emp_weighted_mean).values
)
log.info("OEWS by group: %d groups with employment data", len(oews_by_group))

# Apportion OEWS from SOC 2018 to SOC 2010 across direct crosswalk edges
oews_2018 = oews_lookup[
    ["soc_2018", "group_id", "tot_emp_adjusted", "a_mean", "a_median"]
].drop_duplicates()

oews_edges = xwalk_edges.merge(oews_2018, on="soc_2018", how="left")
n_soc_2010_per_2018 = oews_edges.groupby("soc_2018")["soc_2010"].nunique()
oews_edges["n_soc_2010_per_2018"] = oews_edges["soc_2018"].map(n_soc_2010_per_2018)
oews_edges["oews_tot_emp_allocated"] = (
    oews_edges["tot_emp_adjusted"] / oews_edges["n_soc_2010_per_2018"]
)


def _allocated_emp_weighted_mean(sub):
    """Employment-weighted mean wage using direct-edge allocated employment."""
    mask = sub["a_mean"].notna() & sub["oews_tot_emp_allocated"].notna()
    if not mask.any():
        return np.nan
    return np.average(
        sub.loc[mask, "a_mean"], weights=sub.loc[mask, "oews_tot_emp_allocated"]
    )


oews_by_soc_2010 = (
    oews_edges.groupby("soc_2010")
    .agg(
        oews_tot_emp_allocated=("oews_tot_emp_allocated", "sum"),
        oews_n_soc_2018=("soc_2018", "nunique"),
    )
    .reset_index()
)
oews_by_soc_2010["oews_a_mean"] = (
    oews_edges.groupby("soc_2010").apply(_allocated_emp_weighted_mean).values
)
oews_by_soc_2010 = oews_by_soc_2010.merge(
    soc_2010[["soc_2010", "group_id"]], on="soc_2010", how="left"
)
assert not oews_by_soc_2010["soc_2010"].duplicated().any(), (
    "Expected one allocated OEWS row per SOC 2010 occupation"
)
log.info("OEWS apportioned to SOC 2010: %d occupations", len(oews_by_soc_2010))


def add_task_weights(df):
    """Add equal and employment-based task weights with explicit fallback rules."""
    df = df.copy()

    n_occs_per_task = df.groupby("task_key")["soc_2010"].transform("nunique")
    df["n_occs_per_task"] = n_occs_per_task
    df["equal_task_weight"] = np.where(
        df["n_occs_per_task"] > 0,
        1.0 / df["n_occs_per_task"],
        np.nan,
    )

    task_total_emp = df.groupby("task_key")["oews_tot_emp_imputed"].transform("sum")
    has_positive_emp = task_total_emp > 0
    df["emp_task_weight"] = np.where(
        has_positive_emp,
        df["oews_tot_emp_imputed"] / task_total_emp,
        df["equal_task_weight"],
    )

    equal_weight_sums = df.groupby("task_key")["equal_task_weight"].sum(min_count=1)
    bad_equal = equal_weight_sums[~np.isclose(equal_weight_sums, 1.0, atol=1e-9)]
    assert bad_equal.empty, (
        "Equal task weights do not sum to 1 for tasks: "
        f"{bad_equal.index.tolist()[:10]}"
    )

    emp_weight_sums = df.groupby("task_key")["emp_task_weight"].sum(min_count=1)
    bad_emp = emp_weight_sums[~np.isclose(emp_weight_sums, 1.0, atol=1e-9)]
    assert bad_emp.empty, (
        "Employment task weights do not sum to 1 for tasks: "
        f"{bad_emp.index.tolist()[:10]}"
    )

    return df

no_oews = soc_2018[~soc_2018["soc_2018"].isin(oews_lookup["soc_2018"])]
log.info(
    "OEWS coverage: %d/%d matched (%d exact, %d broad), %d with no OEWS data:",
    len(oews_lookup),
    len(soc_2018),
    len(exact),
    len(broad_matched),
    len(no_oews),
)
for _, row in no_oews.iterrows():
    log.info("  %s  %s", row["soc_2018"], row["title_2018"])

# ======================================================================================
# Step 3: Load Eloundou et al. and merge on SOC 2018
# ======================================================================================
eloundou = pd.read_csv(Path("input") / "eloundou_occ_level.csv")
eloundou["soc_2018"] = eloundou["O*NET-SOC Code"].str[:7]
eloundou_agg = eloundou.groupby("soc_2018").mean(numeric_only=True).reset_index()
log.info(
    "Eloundou: %d raw rows -> %d SOC 2018 codes",
    len(eloundou),
    len(eloundou_agg),
)

df_eloundou, left_only, right_only = merge_with_diagnostics(
    soc_2018, eloundou_agg, on="soc_2018"
)
log_merge_diagnostics(
    left_only,
    right_only,
    left_label="SOC 2018 universe",
    right_label="Eloundou",
    labels=soc_2018,
    key_col="soc_2018",
    logger=log,
)

# Merge OEWS lookup onto Eloundou dataset (drop group_id to avoid duplicate)
df_eloundou = df_eloundou.merge(
    oews_lookup.drop(columns=["group_id"]), on="soc_2018", how="left"
)
df_eloundou["oews_broad_match"] = df_eloundou["broad_match"].fillna(False)
df_eloundou = df_eloundou.rename(
    columns={
        "occ_title": "oews_occ_title",
        "tot_emp": "oews_tot_emp",
        "a_mean": "oews_a_mean",
        "a_median": "oews_a_median",
        "soc_2018_broad": "oews_soc_2018_broad",
        "tot_emp_adjusted": "oews_tot_emp_adjusted",
    }
)
df_eloundou = df_eloundou.drop(columns=["broad_match"])

# Impute missing employment for occupations that have exposure data
has_exposure = df_eloundou["dv_rating_alpha"].notna()
mean_emp = df_eloundou["oews_tot_emp_adjusted"].mean()
median_emp = df_eloundou["oews_tot_emp_adjusted"].median()
log.info(
    "Employment stats (occupations with OEWS employment): mean=%.0f, median=%.0f",
    mean_emp,
    median_emp,
)

df_eloundou["oews_tot_emp_imputed"] = df_eloundou["oews_tot_emp_adjusted"]
missing_emp_with_exposure = has_exposure & df_eloundou["oews_tot_emp_adjusted"].isna()
df_eloundou.loc[missing_emp_with_exposure, "oews_tot_emp_imputed"] = median_emp
log.info(
    "Imputed employment for %d occupations (had exposure but no OEWS)",
    missing_emp_with_exposure.sum(),
)

# Order columns: identifiers, exposure scores, OEWS
exposure_cols = [
    c for c in df_eloundou.columns if c.startswith("dv_") or c.startswith("human_")
]
oews_cols = [c for c in df_eloundou.columns if c.startswith("oews_")]
df_eloundou = df_eloundou[
    ["soc_2018", "title_2018", "group_id"] + exposure_cols + oews_cols
]

# Save checkpoint
df_eloundou.to_csv(OUT / "occupations_eloundou_et_al.csv", index=False)
log.info(
    "Saved checkpoint: %s (%d rows)",
    OUT / "occupations_eloundou_et_al.csv",
    len(df_eloundou),
)

# ======================================================================================
# Step 4: Load AEI data and merge onto O*NET task statements
# ======================================================================================
aei_stacked = load_aei_tasks(logger=log)

# Compute automation and augmentation counts
aei_stacked["automation_count"] = (
    aei_stacked["directive_count"] + aei_stacked["feedback_loop_count"]
)
aei_stacked["augmentation_count"] = (
    aei_stacked["validation_count"]
    + aei_stacked["task_iteration_count"]
    + aei_stacked["learning_count"]
)

# Group by (task_key, platform), sum across releases
aei_by_platform = (
    aei_stacked.groupby(["task_key", "platform"])[
        ["task_count", "automation_count", "augmentation_count"]
    ]
    .sum()
    .reset_index()
)

# Pivot wide: one row per task_key, columns prefixed by platform
aei_wide = aei_by_platform.pivot(index="task_key", columns="platform").reset_index()
aei_wide.columns = [
    f"{col[1]}_{col[0]}" if col[1] else col[0] for col in aei_wide.columns
]
log.info("AEI wide: %d unique tasks", len(aei_wide))

# Merge onto O*NET task-occupation pairs, fill zeros for unobserved tasks
aei_task_occ, left_only, right_only = merge_with_diagnostics(
    task_occ, aei_wide, on="task_key"
)
log_merge_diagnostics(
    left_only,
    right_only,
    left_label="O*NET task-occupations",
    right_label="AEI",
    logger=log,
)
usage_cols = [c for c in aei_task_occ.columns if c.endswith("_count")]
aei_task_occ[usage_cols] = aei_task_occ[usage_cols].fillna(0)

# Combined totals across platforms
for measure in ["task_count", "automation_count", "augmentation_count"]:
    aei_task_occ[f"total_{measure}"] = (
        aei_task_occ[f"claude_ai_{measure}"] + aei_task_occ[f"1p_api_{measure}"]
    )
log.info(
    "AEI task-occupations: %d rows (%d tasks x %d occupations)",
    len(aei_task_occ),
    aei_task_occ["task_key"].nunique(),
    aei_task_occ["soc_2010"].nunique(),
)


# ======================================================================================
# Step 5: Merge AEI task-occupation data onto SOC 2010 universe
# ======================================================================================
df_2010, left_only_2010, right_only_2010 = merge_with_diagnostics(
    soc_2010, aei_task_occ, on="soc_2010"
)
log_merge_diagnostics(
    left_only_2010,
    right_only_2010,
    left_label="SOC 2010 universe",
    right_label="AEI task-occupations",
    labels=soc_2010,
    key_col="soc_2010",
    logger=log,
)

# ======================================================================================
# Step 6: Merge direct-edge apportioned OEWS onto SOC 2010 universe
# ======================================================================================
df_2010 = df_2010.merge(
    oews_by_soc_2010[["soc_2010", "oews_tot_emp_allocated", "oews_a_mean"]],
    on="soc_2010",
    how="left",
)

n_occ_with = df_2010.loc[df_2010["oews_tot_emp_allocated"].notna(), "soc_2010"].nunique()
n_occ_without = df_2010.loc[
    df_2010["oews_tot_emp_allocated"].isna(), "soc_2010"
].nunique()
log.info(
    "SOC 2010 OEWS coverage: %d occupations with employment, %d without",
    n_occ_with,
    n_occ_without,
)

# Impute missing employment with median (for employment-weighted apportionment)
median_emp_2010 = df_2010.loc[
    df_2010["oews_tot_emp_allocated"].notna(), "oews_tot_emp_allocated"
].median()
log.info("SOC 2010 median employment: %.0f", median_emp_2010)

df_2010["oews_tot_emp_imputed"] = df_2010["oews_tot_emp_allocated"].fillna(
    median_emp_2010
)
log.info(
    "Imputed employment for %d occupations",
    n_occ_without,
)

# ======================================================================================
# Step 7: Apportion task counts across occupations and aggregate
# ======================================================================================
count_cols = [c for c in df_2010.columns if c.endswith("_count")]
df_2010 = add_task_weights(df_2010)

for col in count_cols:
    df_2010[f"equal_{col}"] = df_2010[col] * df_2010["equal_task_weight"]

for col in count_cols:
    df_2010[f"emp_{col}"] = df_2010[col] * df_2010["emp_task_weight"]

# Aggregate to occupation level
equal_cols = [f"equal_{col}" for col in count_cols]
emp_cols = [f"emp_{col}" for col in count_cols]

occ_equal = df_2010.groupby("soc_2010")[equal_cols].sum(min_count=1).reset_index()
occ_emp = df_2010.groupby("soc_2010")[emp_cols].sum(min_count=1).reset_index()

# Merge both approaches together with metadata
oews_per_occ = df_2010[
    ["soc_2010", "oews_tot_emp_allocated", "oews_tot_emp_imputed", "oews_a_mean"]
].drop_duplicates("soc_2010")

df_aei = (
    soc_2010[["soc_2010", "title_2010", "group_id"]]
    .merge(oews_per_occ, on="soc_2010", how="left")
    .merge(occ_equal, on="soc_2010", how="left")
    .merge(occ_emp, on="soc_2010", how="left")
)
assert len(df_aei) == len(soc_2010), (
    f"Expected {len(soc_2010)} rows after merge, got {len(df_aei)}"
)

# Per-capita: divide employment-weighted counts by total employment
for col in emp_cols:
    df_aei[f"{col}_pc"] = df_aei[col] / df_aei["oews_tot_emp_imputed"]

# Order columns: identifiers, OEWS, equal, emp, emp per-capita
pc_cols = [f"{col}_pc" for col in emp_cols]
df_aei = df_aei[
    ["soc_2010", "title_2010", "group_id"]
    + ["oews_tot_emp_allocated", "oews_tot_emp_imputed", "oews_a_mean"]
    + equal_cols
    + emp_cols
    + pc_cols
]

log.info(
    "Occupation-level SOC 2010: %d occupations (%d with AEI data)",
    len(df_aei),
    (df_aei["equal_claude_ai_task_count"] > 0).sum(),
)

df_aei.to_csv(OUT / "occupations_aei.csv", index=False)
log.info("Saved checkpoint: %s (%d rows)", OUT / "occupations_aei.csv", len(df_aei))

# ======================================================================================
# Step 8: Load AEI occupation automation/augmentation data (2025-03-27 release)
# ======================================================================================
aei_occ_raw = pd.read_csv(
    Path("input") / "aei_occupation_automation_augmentation_data.csv"
)
aei_occ_raw["soc_2010"] = aei_occ_raw["O*NET-SOC Code"].str[:7]

# Unlike task-level AEI data (where counts are additive and get summed in step 7),
# the auto/aug data contains pre-computed ratios keyed on 8-digit O*NET-SOC codes.
# pct_occ_scaled is additive (share of usage), so we sum it.
# Ratios are weighted by pct_occ_scaled when collapsing to 6-digit SOC 2010.
aei_occ_cols = [
    "pct_occ_scaled",
    "augmentation_weighted_ratio",
    "automation_weighted_ratio",
]


def _pct_weighted_mean(sub):
    """Weighted mean of ratios using pct_occ_scaled as weights."""
    w = sub["pct_occ_scaled"]
    if w.sum() == 0:
        return sub[["augmentation_weighted_ratio", "automation_weighted_ratio"]].mean()
    return pd.Series(
        {
            "augmentation_weighted_ratio": np.average(
                sub["augmentation_weighted_ratio"], weights=w
            ),
            "automation_weighted_ratio": np.average(
                sub["automation_weighted_ratio"], weights=w
            ),
        }
    )


aei_occ_agg = aei_occ_raw.groupby("soc_2010").apply(_pct_weighted_mean).reset_index()
aei_occ_agg["pct_occ_scaled"] = (
    aei_occ_raw.groupby("soc_2010")["pct_occ_scaled"].sum().values
)
log.info(
    "AEI occupation auto/aug: %d raw rows -> %d SOC 2010 codes",
    len(aei_occ_raw),
    len(aei_occ_agg),
)

# Merge onto SOC 2010 universe with OEWS data
df_aei_occ, left_only, right_only = merge_with_diagnostics(
    soc_2010, aei_occ_agg, on="soc_2010"
)
log_merge_diagnostics(
    left_only,
    right_only,
    left_label="SOC 2010 universe",
    right_label="AEI auto/aug",
    labels=soc_2010,
    key_col="soc_2010",
    logger=log,
)

# Add direct-edge apportioned OEWS data
df_aei_occ = df_aei_occ.merge(
    oews_by_soc_2010[["soc_2010", "oews_tot_emp_allocated", "oews_a_mean"]],
    on="soc_2010",
    how="left",
)
df_aei_occ["oews_tot_emp_imputed"] = df_aei_occ["oews_tot_emp_allocated"].fillna(
    median_emp_2010
)

# Zero-recode: occupations with O*NET tasks but not in this file get 0
has_tasks = df_aei_occ["soc_2010"].isin(task_occ_occupations)
for col in aei_occ_cols:
    df_aei_occ.loc[has_tasks & df_aei_occ[col].isna(), col] = 0

n_with_data = df_aei_occ["pct_occ_scaled"].gt(0).sum()
n_zero = (df_aei_occ["pct_occ_scaled"] == 0).sum()
n_na = df_aei_occ["pct_occ_scaled"].isna().sum()
log.info(
    "AEI auto/aug coverage: %d with data, %d zero-recoded, %d missing (no O*NET tasks)",
    n_with_data,
    n_zero,
    n_na,
)

# Per-capita usage: scale pct_occ_scaled by employment
df_aei_occ["pct_occ_scaled_pc"] = (
    df_aei_occ["pct_occ_scaled"] / df_aei_occ["oews_tot_emp_imputed"]
)

# Order columns: identifiers, OEWS, ratios, per-capita
df_aei_occ = df_aei_occ[
    ["soc_2010", "title_2010", "group_id"]
    + ["oews_tot_emp_allocated", "oews_tot_emp_imputed", "oews_a_mean"]
    + aei_occ_cols
    + ["pct_occ_scaled_pc"]
]

df_aei_occ.to_csv(OUT / "occupations_aei_auto_aug_2025_03_27.csv", index=False)
log.info(
    "Saved: %s (%d rows)",
    OUT / "occupations_aei_auto_aug_2025_03_27.csv",
    len(df_aei_occ),
)
