"""Read the downloaded Claude.ai AEI CSV and summarize unique column values."""

from pathlib import Path

import pandas as pd

RELEASE = "release_2026_03_24"
CSV = (
    Path(__file__).resolve().parent.parent
    / "data"
    / RELEASE
    / "aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv"
)

df = pd.read_csv(CSV)

for col in df.columns:
    if col == "value":
        continue
    uniques = df[col].unique()
    print(f"\n=== {col} ({len(uniques)} unique) ===")
    print(uniques)

print("\n=== head ===")
print(df.head())

before = len(df)
df = df[df["geo_id"].isin(["USA", "GLOBAL"])]
after = len(df)
print(f"\nfiltered geo_id in (USA, GLOBAL): {before} -> {after} rows")

onet_facets = df.loc[df["facet"].str.contains("onet_task", na=False), "facet"].unique()
print(f"\n=== facet values containing 'onet_task' ({len(onet_facets)}) ===")
print(onet_facets)

before = len(df)
df = df[df["facet"] == "onet_task::collaboration"]
after = len(df)
print(f"\nfiltered facet == onet_task::collaboration: {before} -> {after} rows")

print("\n=== head ===")
print(df.head())

for col in df.columns:
    if col == "value":
        continue
    uniques = df[col].unique()
    print(f"\n=== {col} ({len(uniques)} unique) ===")
    print(uniques)
