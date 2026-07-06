"""Convert O*NET Excel files to CSV in data/onet/."""

from pathlib import Path

import pandas as pd

from eai.codebook import update_codebook

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
xwalk_src = INPUT / "soc_2010_to_2018_crosswalk.xlsx"
xwalk_dest = OUT / "soc_2010_to_2018_crosswalk.csv"
xwalk = pd.read_excel(xwalk_src, header=7)
xwalk.columns = ["soc_2010", "title_2010", "soc_2018", "title_2018"]
xwalk.to_csv(xwalk_dest, index=False)
print(f"wrote {len(xwalk)} rows -> {xwalk_dest}")

# --- Codebook ---
update_codebook(
    OUT / "codebook.md",
    section="onet_reference",
    title="O*NET reference tables",
    source="pipeline/convert_onet.py",
    files=[
        {
            "name": "task_statements.csv",
            "description": (
                "O*NET 20.1 Task Statements workbook converted to CSV with "
                "columns unchanged."
            ),
            "columns": [
                ("O*NET-SOC Code", "Eight-character O*NET-SOC occupation code."),
                ("Title", "O*NET occupation title."),
                ("Task ID", "O*NET task identifier."),
                ("Task", "Task statement text."),
                (
                    "Task Type",
                    "Core or Supplemental; blank when the task is unclassified.",
                ),
                (
                    "Incumbents Responding",
                    "Number of survey incumbents rating the task.",
                ),
                ("Date", "Task update date (mm/YYYY)."),
                ("Domain Source", "O*NET data collection source for the task."),
            ],
        },
        {
            "name": "soc_2010_to_2018_crosswalk.csv",
            "description": (
                "BLS SOC 2010 to SOC 2018 crosswalk workbook converted to "
                "CSV. Contains one stray repeated-header row, which "
                "pipeline/build_crosswalk.py drops."
            ),
            "columns": [
                ("soc_2010", "Six-digit SOC 2010 occupation code."),
                ("title_2010", "SOC 2010 occupation title."),
                ("soc_2018", "Six-digit SOC 2018 occupation code."),
                ("title_2018", "SOC 2018 occupation title."),
            ],
        },
    ],
)
print(f"updated codebook -> {OUT / 'codebook.md'}")
