"""Convert OEWS Excel files to CSV with lowercase column names."""

from pathlib import Path

import pandas as pd

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
