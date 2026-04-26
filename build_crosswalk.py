"""Build SOC 2010→2018 crosswalk artifacts."""

from pathlib import Path

import networkx as nx
import pandas as pd

OUT = Path("output") / "onet"
XWALK_CSV = OUT / "soc_2010_to_2018_crosswalk.csv"
EDGES_CSV = OUT / "soc_crosswalk_edges.csv"

# --- Load crosswalk, drop stray header row ---
xwalk = pd.read_csv(XWALK_CSV)
xwalk = xwalk[xwalk["soc_2010"] != "2010 SOC Code"].copy()

# --- Build clean edge list ---
xwalk_edges = (
    xwalk[["soc_2010", "title_2010", "soc_2018", "title_2018"]]
    .drop_duplicates()
    .sort_values(["soc_2018", "soc_2010"])
    .reset_index(drop=True)
)

# --- Build bipartite graph and find connected components ---
G = nx.Graph()
for _, row in xwalk_edges.iterrows():
    node_2010 = ("2010", row["soc_2010"])
    node_2018 = ("2018", row["soc_2018"])
    G.add_node(node_2010, title=row["title_2010"])
    G.add_node(node_2018, title=row["title_2018"])
    G.add_edge(node_2010, node_2018)

components = list(nx.connected_components(G))
print(f"{len(components)} connected components from {len(G.nodes)} nodes")

# --- Assign group IDs ---
rows_2010 = []
rows_2018 = []
for group_id, component in enumerate(components):
    for vintage, code in component:
        title = G.nodes[(vintage, code)]["title"]
        if vintage == "2010":
            rows_2010.append(
                {"soc_2010": code, "title_2010": title, "group_id": group_id}
            )
        else:
            rows_2018.append(
                {"soc_2018": code, "title_2018": title, "group_id": group_id}
            )

df_2010 = pd.DataFrame(rows_2010).sort_values("soc_2010").reset_index(drop=True)
df_2018 = pd.DataFrame(rows_2018).sort_values("soc_2018").reset_index(drop=True)

# --- Report non-trivial components ---
n_trivial = sum(1 for c in components if len(c) == 2)
n_nontrivial = len(components) - n_trivial
print(f"{n_trivial} trivial (1:1) components, {n_nontrivial} non-trivial")

for group_id, component in enumerate(components):
    if len(component) > 2:
        codes_2010 = sorted(c for v, c in component if v == "2010")
        codes_2018 = sorted(c for v, c in component if v == "2018")
        print(f"  group {group_id}: {codes_2010} <--> {codes_2018}")

# --- Write ---
dest_2010 = OUT / "soc_2010_to_group.csv"
dest_2018 = OUT / "soc_2018_to_group.csv"
xwalk_edges.to_csv(EDGES_CSV, index=False)
df_2010.to_csv(dest_2010, index=False)
df_2018.to_csv(dest_2018, index=False)
print(f"wrote {len(xwalk_edges)} rows -> {EDGES_CSV}")
print(f"\nwrote {len(df_2010)} rows -> {dest_2010}")
print(f"wrote {len(df_2018)} rows -> {dest_2018}")
