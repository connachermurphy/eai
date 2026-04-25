"""Build an OEWS panel across years and analyze occupational footprint changes."""

from pathlib import Path

import pandas as pd

from eai.utils import get_logger

log = get_logger(__name__)
DATA = Path("output")

# ======================================================================================
# Step 1: Load and stack OEWS files into a panel
# ======================================================================================
OEWS_FILES = sorted(DATA.glob("oews/national_M*_dl.csv"))
log.info("Found %d OEWS files: %s", len(OEWS_FILES), [f.name for f in OEWS_FILES])

frames = []
for path in OEWS_FILES:
    year = int(path.stem.split("_")[1].lstrip("M"))
    raw = pd.read_csv(path)
    detailed = raw[raw["o_group"] == "detailed"][
        ["occ_code", "occ_title", "tot_emp", "a_mean", "a_median"]
    ].copy()
    detailed["year"] = year
    for col in ["tot_emp", "a_mean", "a_median"]:
        detailed[col] = pd.to_numeric(detailed[col], errors="coerce")
    frames.append(detailed)
    log.info("  %d: %d detailed occupations", year, len(detailed))

panel = pd.concat(frames, ignore_index=True)
log.info("Panel: %d rows (%d years x ~%d occupations)", len(panel), panel["year"].nunique(), len(panel) // panel["year"].nunique())

# ======================================================================================
# Step 2: Balanced panel indicator and footprint changes
# ======================================================================================
years = sorted(panel["year"].unique())
years_per_code = panel.groupby("occ_code")["year"].nunique()
balanced_codes = set(years_per_code[years_per_code == len(years)].index)
panel["balanced_panel"] = panel["occ_code"].isin(balanced_codes)

unbalanced = panel.loc[~panel["balanced_panel"], ["occ_code", "occ_title", "year"]].drop_duplicates()
log.info(
    "Balanced panel: %d codes in all %d years, %d unbalanced:",
    len(balanced_codes), len(years), panel["occ_code"].nunique() - len(balanced_codes),
)
for code, grp in unbalanced.groupby("occ_code"):
    title = grp["occ_title"].iloc[0]
    present_years = sorted(grp["year"])
    log.info("  %s  %s  (present: %s)", code, title, present_years)
