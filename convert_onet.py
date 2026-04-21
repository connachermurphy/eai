"""Convert O*NET Excel files to CSV in data/onet/."""

from pathlib import Path

import pandas as pd

INPUT = Path("input")
OUT = Path("output") / "onet"
OUT.mkdir(parents=True, exist_ok=True)

# --- Task Statements ---
task_src = INPUT / "Task Statements.xlsx"
task_dest = OUT / "task_statements.csv"
tasks = pd.read_excel(task_src)
tasks.to_csv(task_dest, index=False)
print(f"wrote {len(tasks)} rows -> {task_dest}")

# --- SOC 2010→2018 Crosswalk ---
xwalk_src = INPUT / "soc_2010_to_2018_crosswalk .xlsx"
xwalk_dest = OUT / "soc_2010_to_2018_crosswalk.csv"
xwalk = pd.read_excel(xwalk_src, header=7)
xwalk.columns = ["soc_2010", "title_2010", "soc_2018", "title_2018"]
xwalk.to_csv(xwalk_dest, index=False)
print(f"wrote {len(xwalk)} rows -> {xwalk_dest}")
