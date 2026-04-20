"""Convert OEWS Excel files to CSV with lowercase column names."""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OEWS_DIR = Path(__file__).resolve().parent

FILES = [
    {
        "src": BASE / "data" / "oesm24all" / "all_data_M_2024.xlsx",
        "sheet": "All May 2024 data",
        "dest": BASE / "data" / "oesm24all" / "all_data_M_2024.csv",
    },
    {
        "src": OEWS_DIR / "national_M2021_dl.xlsx",
        "sheet": "national_M2021_dl",
        "dest": BASE / "data" / "oews" / "national_M2021_dl.csv",
    },
    {
        "src": OEWS_DIR / "national_M2022_dl.xlsx",
        "sheet": "national_M2022_dl",
        "dest": BASE / "data" / "oews" / "national_M2022_dl.csv",
    },
    {
        "src": OEWS_DIR / "national_M2023_dl.xlsx",
        "sheet": "national_M2023_dl",
        "dest": BASE / "data" / "oews" / "national_M2023_dl.csv",
    },
    {
        "src": OEWS_DIR / "national_M2024_dl.xlsx",
        "sheet": "national_M2024_dl",
        "dest": BASE / "data" / "oews" / "national_M2024_dl.csv",
    },
]

for f in FILES:
    f["dest"].parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(f["src"], sheet_name=f["sheet"], dtype=str)
    df.columns = df.columns.str.lower()
    df.to_csv(f["dest"], index=False)
    print(f"wrote {len(df):,} rows -> {f['dest']}")
