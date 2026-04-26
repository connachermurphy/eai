"""Build an OEWS panel across years and analyze occupational footprint changes."""

from pathlib import Path

import pandas as pd

from eai.utils import get_logger

log = get_logger(__name__)
DATA = Path("output")
OUT = Path("output")

# ======================================================================================
# Step 1: Load and stack OEWS files into a panel
# ======================================================================================
OEWS_YEARS = range(2021, 2025)
OEWS_FILES = [DATA / "oews" / f"national_M{y}_dl.csv" for y in OEWS_YEARS]
log.info("Found %d OEWS files: %s", len(OEWS_FILES), [f.name for f in OEWS_FILES])

frames = []
for path in OEWS_FILES:
    year = int(path.stem.split("_")[1].lstrip("M"))
    raw = pd.read_csv(path)
    KEEP_COLS = [
        "occ_code",
        "occ_title",
        "tot_emp",
        "emp_prse",
        "a_mean",
        "mean_prse",
        "a_pct10",
        "a_pct25",
        "a_median",
        "a_pct75",
        "a_pct90",
    ]
    detailed = raw[raw["o_group"] == "detailed"][KEEP_COLS].copy()
    detailed["year"] = year
    NUMERIC_COLS = [c for c in KEEP_COLS if c not in ("occ_code", "occ_title")]
    for col in NUMERIC_COLS:
        detailed[col] = pd.to_numeric(detailed[col], errors="coerce")
    frames.append(detailed)
    log.info("  %d: %d detailed occupations", year, len(detailed))

panel = pd.concat(frames, ignore_index=True)
log.info(
    "Panel: %d rows (%d years x ~%d occupations)",
    len(panel),
    panel["year"].nunique(),
    len(panel) // panel["year"].nunique(),
)

# ======================================================================================
# Step 2: Balanced panel indicator and footprint changes
# ======================================================================================
years = sorted(panel["year"].unique())
years_per_code = panel.groupby("occ_code")["year"].nunique()
balanced_codes = set(years_per_code[years_per_code == len(years)].index)
panel["balanced_panel"] = panel["occ_code"].isin(balanced_codes)

unbalanced = panel.loc[
    ~panel["balanced_panel"], ["occ_code", "occ_title", "year"]
].drop_duplicates()
log.info(
    "Balanced panel: %d codes in all %d years, %d unbalanced:",
    len(balanced_codes),
    len(years),
    panel["occ_code"].nunique() - len(balanced_codes),
)
for code, grp in unbalanced.groupby("occ_code"):
    title = grp["occ_title"].iloc[0]
    present_years = sorted(grp["year"])
    log.info("  %s  %s  (present: %s)", code, title, present_years)

panel.to_csv(OUT / "oews_panel.csv", index=False)
log.info("Saved: %s (%d rows)", OUT / "oews_panel.csv", len(panel))
