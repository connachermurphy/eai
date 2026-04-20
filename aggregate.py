"""Aggregate task-level merged data to occupation-level summaries.

Produces occupation-level (SOC 2010) usage by collaboration mode under two
apportionment approaches:
  1. Equal weighting: divide task usage by number of occupations sharing the task
  2. Employment weighting: divide proportional to occupation employment

Saves checkpoints at each stage.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
MERGED_CSV = BASE / "data" / "merged.csv"
OUT_DIR = BASE / "data"

COLLAB_TYPES = [
    "directive",
    "feedback_loop",
    "learning",
    "none",
    "not_classified",
    "task_iteration",
    "validation",
]
COUNT_COLS = [f"{t}_count" for t in COLLAB_TYPES]

# ==========================================================================
# Load
# ==========================================================================
df = pd.read_csv(MERGED_CSV)
df["task_key"] = df["Task"].str.lower().str.strip()
logger.info("Loaded %d task-occupation rows", len(df))
logger.info("  Unique tasks: %d", df["task_key"].nunique())
logger.info("  Unique SOC 2010 occupations: %d", df["soc_2010"].nunique())

# ==========================================================================
# Approach 1: Equal weighting
# ==========================================================================
logger.info("--- Approach 1: Equal weighting ---")

# Count how many occupations each task maps to
n_occs_per_task = df.groupby("task_key")["soc_2010"].nunique().rename("n_occs_per_task")
df = df.merge(n_occs_per_task, on="task_key", how="left")

# Apportion counts equally across occupations
for col in ["task_count"] + COUNT_COLS:
    df[f"equal_{col}"] = df[col] / df["n_occs_per_task"]

# Aggregate to occupation level
equal_cols = [f"equal_{col}" for col in ["task_count"] + COUNT_COLS]
occ_equal = df.groupby("soc_2010")[equal_cols].sum().reset_index()

# Compute collaboration ratios
for t in COLLAB_TYPES:
    occ_equal[f"{t}_ratio"] = (
        occ_equal[f"equal_{t}_count"] / occ_equal["equal_task_count"]
    )

# Automation and augmentation
occ_equal["augmentation_ratio"] = (
    occ_equal["validation_ratio"]
    + occ_equal["task_iteration_ratio"]
    + occ_equal["learning_ratio"]
)
occ_equal["automation_ratio"] = (
    occ_equal["directive_ratio"] + occ_equal["feedback_loop_ratio"]
)

# Add employment and wage data
emp_cols = ["soc_2010", "apportioned_occ_emp", "coarsened_group_a_mean"]
emp = df[emp_cols].drop_duplicates("soc_2010")
occ_equal = occ_equal.merge(emp, on="soc_2010", how="left")

logger.info("  Occupations: %d", len(occ_equal))
logger.info("  With AEI data: %d", (occ_equal["equal_task_count"] > 0).sum())
total_equal = occ_equal["equal_task_count"].sum()
logger.info("  Total apportioned task_count: %s", f"{total_equal:,.0f}")

occ_equal.to_csv(OUT_DIR / "occ_equal_weighted.csv", index=False)
logger.info("  Saved to %s", OUT_DIR / "occ_equal_weighted.csv")

# ==========================================================================
# Approach 2: Employment weighting
# ==========================================================================
logger.info("--- Approach 2: Employment weighting ---")

# Total employment across all occupations sharing each task
task_total_emp = (
    df[df["apportioned_occ_emp"].notna()]
    .groupby("task_key")["apportioned_occ_emp"]
    .sum()
    .rename("task_total_emp")
)
df = df.merge(task_total_emp, on="task_key", how="left")

# Apportion counts by employment share
for col in ["task_count"] + COUNT_COLS:
    df[f"emp_{col}"] = df[col]  # default: undivided
    mask = df["apportioned_occ_emp"].notna() & (df["task_total_emp"] > 0)
    df.loc[mask, f"emp_{col}"] = (
        df.loc[mask, col]
        * df.loc[mask, "apportioned_occ_emp"]
        / df.loc[mask, "task_total_emp"]
    )

# Aggregate to occupation level
emp_weight_cols = [f"emp_{col}" for col in ["task_count"] + COUNT_COLS]
occ_emp = df.groupby("soc_2010")[emp_weight_cols].sum().reset_index()

# Compute collaboration ratios
for t in COLLAB_TYPES:
    occ_emp[f"{t}_ratio"] = occ_emp[f"emp_{t}_count"] / occ_emp["emp_task_count"]

# Automation and augmentation
occ_emp["augmentation_ratio"] = (
    occ_emp["validation_ratio"]
    + occ_emp["task_iteration_ratio"]
    + occ_emp["learning_ratio"]
)
occ_emp["automation_ratio"] = (
    occ_emp["directive_ratio"] + occ_emp["feedback_loop_ratio"]
)

# Add employment and wage data
occ_emp = occ_emp.merge(emp, on="soc_2010", how="left")

logger.info("  Occupations: %d", len(occ_emp))
logger.info("  With AEI data: %d", (occ_emp["emp_task_count"] > 0).sum())
total_emp_weighted = occ_emp["emp_task_count"].sum()
logger.info("  Total apportioned task_count: %s", f"{total_emp_weighted:,.0f}")

occ_emp.to_csv(OUT_DIR / "occ_emp_weighted.csv", index=False)
logger.info("  Saved to %s", OUT_DIR / "occ_emp_weighted.csv")

# ==========================================================================
# Comparison
# ==========================================================================
logger.info("--- Comparison ---")
equal_compare_cols = [
    "soc_2010",
    "equal_task_count",
    "automation_ratio",
    "augmentation_ratio",
]
emp_compare_cols = [
    "soc_2010",
    "emp_task_count",
    "automation_ratio",
    "augmentation_ratio",
]
both = occ_equal[equal_compare_cols].merge(
    occ_emp[emp_compare_cols],
    on="soc_2010",
    suffixes=("_equal", "_emp"),
)
has_data = both[(both["equal_task_count"] > 0) & (both["emp_task_count"] > 0)]
auto_corr = has_data["automation_ratio_equal"].corr(has_data["automation_ratio_emp"])
aug_corr = has_data["augmentation_ratio_equal"].corr(has_data["augmentation_ratio_emp"])
logger.info("  Occupations with data in both: %d", len(has_data))
logger.info("  Automation ratio correlation: %.4f", auto_corr)
logger.info("  Augmentation ratio correlation: %.4f", aug_corr)
