"""Convert OEWS Excel files to CSV with lowercase column names."""

from pathlib import Path

import pandas as pd

from eai.codebook import update_codebook

INPUT = Path("input")
OUT = Path("output")

FILES = [
    {
        "src": INPUT / "all_data_M_2024.xlsx",
        "sheet": "All May 2024 data",
        "dest": OUT / "oesm24all" / "all_data_M_2024.csv",
    },
    {
        "src": INPUT / "national_M2019_dl.xlsx",
        "sheet": "national_M2019_dl",
        "dest": OUT / "oews" / "national_M2019_dl.csv",
    },
    {
        "src": INPUT / "national_M2020_dl.xlsx",
        "sheet": "national_M2020_dl",
        "dest": OUT / "oews" / "national_M2020_dl.csv",
    },
    {
        "src": INPUT / "national_M2021_dl.xlsx",
        "sheet": "national_M2021_dl",
        "dest": OUT / "oews" / "national_M2021_dl.csv",
    },
    {
        "src": INPUT / "national_M2022_dl.xlsx",
        "sheet": "national_M2022_dl",
        "dest": OUT / "oews" / "national_M2022_dl.csv",
    },
    {
        "src": INPUT / "national_M2023_dl.xlsx",
        "sheet": "national_M2023_dl",
        "dest": OUT / "oews" / "national_M2023_dl.csv",
    },
    {
        "src": INPUT / "national_M2024_dl.xlsx",
        "sheet": "national_M2024_dl",
        "dest": OUT / "oews" / "national_M2024_dl.csv",
    },
]

for f in FILES:
    f["dest"].parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(f["src"], sheet_name=f["sheet"], dtype=str)
    df.columns = df.columns.str.lower()
    df.to_csv(f["dest"], index=False)
    print(f"wrote {len(df):,} rows -> {f['dest']}")

# --- Codebooks ---
# BLS field definitions with lowercased names; values are kept as reported
# (including '*'/'**' suppression markers), so downstream scripts coerce to
# numeric themselves. See the BLS OEWS documentation for full detail:
# https://www.bls.gov/oes/oes_doc.htm
OEWS_COLUMNS = [
    ("area", "BLS area code (99 for U.S. national files)."),
    ("area_title", "Area name."),
    ("area_type", "Area type code (1 = national)."),
    (
        "prim_state",
        "Primary state for the area; absent from some older national files.",
    ),
    ("naics", "NAICS industry code for the estimate."),
    ("naics_title", "NAICS industry title."),
    ("i_group", "Industry level of the estimate (e.g., cross-industry)."),
    ("own_code", "Ownership type code."),
    ("occ_code", "Six-digit SOC occupation code (SOC 2018 in recent years)."),
    ("occ_title", "SOC occupation title."),
    (
        "o_group",
        "SOC aggregation level of the row: total, major, minor, broad, or detailed.",
    ),
    ("tot_emp", "Estimated total employment (excludes self-employed)."),
    ("emp_prse", "Percent relative standard error of employment."),
    ("jobs_1000", "Jobs per 1,000 in the area (blank in national files)."),
    ("loc_quotient", "Location quotient (blank in national files)."),
    ("pct_total", "Percent of industry employment in the occupation."),
    (
        "pct_rpt",
        "Percent of establishments reporting the occupation; absent from some "
        "older files.",
    ),
    ("h_mean", "Mean hourly wage."),
    ("a_mean", "Mean annual wage."),
    ("mean_prse", "Percent relative standard error of the mean wage."),
    ("h_pct10", "10th-percentile hourly wage."),
    ("h_pct25", "25th-percentile hourly wage."),
    ("h_median", "Median hourly wage."),
    ("h_pct75", "75th-percentile hourly wage."),
    ("h_pct90", "90th-percentile hourly wage."),
    ("a_pct10", "10th-percentile annual wage."),
    ("a_pct25", "25th-percentile annual wage."),
    ("a_median", "Median annual wage."),
    ("a_pct75", "75th-percentile annual wage."),
    ("a_pct90", "90th-percentile annual wage."),
    (
        "annual",
        "True when only an annual wage is published (no hourly equivalent).",
    ),
    (
        "hourly",
        "True when only an hourly wage is published (no annual equivalent).",
    ),
]

update_codebook(
    OUT / "oews" / "codebook.md",
    section="oews_national",
    title="OEWS national files",
    source="pipeline/convert_oews.py",
    files=[
        {
            "name": "national_M{year}_dl.csv (2019-2024)",
            "description": (
                "BLS OEWS national occupational employment and wage files, "
                "converted from Excel with lowercased column names and values "
                "unchanged."
            ),
            "columns": OEWS_COLUMNS,
        }
    ],
)
update_codebook(
    OUT / "oesm24all" / "codebook.md",
    section="oews_all_2024",
    title="OEWS May 2024 all-data file",
    source="pipeline/convert_oews.py",
    files=[
        {
            "name": "all_data_M_2024.csv",
            "description": (
                "BLS OEWS May 2024 all-areas, all-industries file, converted "
                "from Excel with lowercased column names and values unchanged."
            ),
            "columns": OEWS_COLUMNS,
        }
    ],
)
print("updated codebooks -> output/oews, output/oesm24all")
