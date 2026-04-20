"""Create merged occupational characteristics datasets."""

from pathlib import Path

import pandas as pd

from eai.utils import get_logger, log_merge_diagnostics, merge_with_diagnostics

log = get_logger(__name__)
DATA = Path("data")

# ========================================================================================
# Step 1: Load the SOC 2018 universe from the connected-component crosswalk
# ========================================================================================
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

# ========================================================================================
# Step 2: Load Eloundou et al. and merge on SOC 2018
# ========================================================================================
eloundou = pd.read_csv(Path("eloundou_et_al") / "occ_level.csv")
eloundou["soc_2018"] = eloundou["O*NET-SOC Code"].str[:7]
eloundou_agg = eloundou.groupby("soc_2018").mean(numeric_only=True).reset_index()
log.info(
    "Eloundou: %d raw rows -> %d SOC 2018 codes",
    len(eloundou),
    len(eloundou_agg),
)

df, left_only, right_only = merge_with_diagnostics(
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

# ========================================================================================
# Step 3: Load 2022 OEWS data and merge on SOC 2018 (exact + broad)
# ========================================================================================
oews_raw = pd.read_csv(DATA / "oews" / "national_M2022_dl.csv")
oews = oews_raw[oews_raw["o_group"] == "detailed"][
    ["occ_code", "occ_title", "tot_emp", "a_mean", "a_median"]
].copy()
oews = oews.rename(columns={"occ_code": "soc_2018"})
for col in ["tot_emp", "a_mean", "a_median"]:
    oews[col] = pd.to_numeric(oews[col], errors="coerce")
log.info("OEWS 2022: %d detailed occupations", len(oews))

# Build a lookup keyed on soc_2018 that covers all universe codes possible:
# 1) Exact matches
# 2) Broad matches for the rest (coarsen last digit -> 0)
universe_codes = set(df["soc_2018"])
oews_codes = set(oews["soc_2018"])

exact = oews[oews["soc_2018"].isin(universe_codes)].copy()
exact["broad_match"] = False
exact["soc_2018_broad"] = pd.NA

unmatched_codes = universe_codes - oews_codes
unmatched_oews = oews[oews["soc_2018"].isin(oews_codes - universe_codes)]

# Coarsen universe codes (last digit -> 0) and match against literal OEWS codes
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

# Combine exact + broad into a single OEWS lookup
exact["tot_emp_adjusted"] = exact["tot_emp"]
oews_lookup = pd.concat([
    exact[["soc_2018", "occ_title", "tot_emp", "a_mean", "a_median", "broad_match", "soc_2018_broad", "tot_emp_adjusted"]],
    broad_matched[["soc_2018", "occ_title", "tot_emp", "a_mean", "a_median", "broad_match", "soc_2018_broad", "tot_emp_adjusted"]],
])
log.info(
    "OEWS lookup: %d exact + %d broad = %d total",
    len(exact),
    len(broad_matched),
    len(oews_lookup),
)

# Single merge onto df
df = df.merge(oews_lookup, on="soc_2018", how="left")
df["broad_match"] = df["broad_match"].fillna(False)

no_oews = df[df["tot_emp"].isna()]
log.info(
    "OEWS coverage: %d/%d matched (%d exact, %d broad), %d with no OEWS data:",
    len(df) - len(no_oews),
    len(df),
    len(exact),
    len(broad_matched),
    len(no_oews),
)
for _, row in no_oews.iterrows():
    log.info("  %s  %s", row["soc_2018"], row["title_2018"])

# Impute missing employment for occupations that have exposure data
has_exposure = df["dv_rating_alpha"].notna()
mean_emp = df.loc[has_exposure, "tot_emp_adjusted"].mean()
median_emp = df.loc[has_exposure, "tot_emp_adjusted"].median()
log.info(
    "Employment stats (occupations with exposure): mean=%.0f, median=%.0f",
    mean_emp,
    median_emp,
)

df["tot_emp_imputed"] = df["tot_emp_adjusted"]
missing_emp_with_exposure = has_exposure & df["tot_emp_adjusted"].isna()
df.loc[missing_emp_with_exposure, "tot_emp_imputed"] = median_emp
log.info(
    "Imputed employment for %d occupations (had exposure but no OEWS)",
    missing_emp_with_exposure.sum(),
)

# Save checkpoint
OUT = Path("occupations")
df.to_csv(OUT / "occupations_eloundou_et_al.csv", index=False)
log.info("Saved checkpoint: %s (%d rows)", OUT / "occupations_eloundou_et_al.csv", len(df))

# ========================================================================================
# Step 4: Load the SOC 2010 universe from the connected-component crosswalk
# ========================================================================================
soc_2010 = pd.read_csv(DATA / "onet" / "soc_2010_to_group.csv")
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
