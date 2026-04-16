"""Convert the O*NET Task Statements xlsx to CSV in data/onet/."""

from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent / "Task Statements.xlsx"
OUT = Path(__file__).resolve().parent.parent / "data" / "onet"
OUT.mkdir(parents=True, exist_ok=True)
DEST = OUT / "task_statements.csv"

df = pd.read_excel(SRC)
df.to_csv(DEST, index=False)
print(f"wrote {len(df)} rows -> {DEST}")
