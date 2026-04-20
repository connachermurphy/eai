"""Create merged occupational characteristics datasets."""

from pathlib import Path

import pandas as pd

from eai.utils import merge_with_diagnostics

DATA = Path("data")

# ========================================================================================
# Step 1: Load the SOC 2018 universe from the connected-component crosswalk
# ========================================================================================
soc_2018 = pd.read_csv(DATA / "onet" / "soc_2018_to_group.csv")
print(f"SOC 2018 universe: {len(soc_2018)} occupations, {soc_2018['group_id'].nunique()} groups")

# ========================================================================================
# Step 2: Load Eloundou et al. and merge on SOC 2018
# ========================================================================================
eloundou = pd.read_csv(Path("eloundou_et_al") / "occ_level.csv")
eloundou["soc_2018"] = eloundou["O*NET-SOC Code"].str[:7]
eloundou_agg = eloundou.groupby("soc_2018").mean(numeric_only=True).reset_index()
print(f"Eloundou: {len(eloundou)} raw rows -> {len(eloundou_agg)} SOC 2018 codes")

df = merge_with_diagnostics(
    soc_2018,
    eloundou_agg,
    on="soc_2018",
    left_label="SOC 2018 universe",
    right_label="Eloundou",
)
