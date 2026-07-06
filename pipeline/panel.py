"""Build an OEWS panel across years with a balanced-panel indicator."""

from pathlib import Path

import pandas as pd

from eai.codebook import update_codebook
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
# Step 2: Balanced panel indicator
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

# ======================================================================================
# Step 3: Update the output codebook
# ======================================================================================
update_codebook(
    OUT / "codebook.md",
    section="oews_panel",
    title="OEWS year panel",
    source="pipeline/panel.py",
    files=[
        {
            "name": "oews_panel.csv",
            "description": (
                "Stacked OEWS national detailed occupations for 2021-2024, "
                "with numeric fields coerced (suppression markers become NA)."
            ),
            "columns": [
                ("occ_code", "Six-digit SOC 2018 occupation code."),
                ("occ_title", "SOC occupation title."),
                ("tot_emp", "Estimated total employment."),
                ("emp_prse", "Percent relative standard error of employment."),
                ("a_mean", "Mean annual wage."),
                ("mean_prse", "Percent relative standard error of the mean wage."),
                ("a_pct10", "10th-percentile annual wage."),
                ("a_pct25", "25th-percentile annual wage."),
                ("a_median", "Median annual wage."),
                ("a_pct75", "75th-percentile annual wage."),
                ("a_pct90", "90th-percentile annual wage."),
                ("year", "OEWS reference year (May estimates)."),
                (
                    "balanced_panel",
                    "True when the occupation code appears in every year of the panel.",
                ),
            ],
        }
    ],
)
log.info("Updated codebook: %s", OUT / "codebook.md")
