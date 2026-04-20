import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("anthropic/occupation_automation_augmentation_data.csv")

df["augmentation_scaled"] = df["augmentation_weighted_ratio"] * df["pct_occ_scaled"]
df["automation_scaled"] = df["automation_weighted_ratio"] * df["pct_occ_scaled"]

fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(
    df["augmentation_scaled"],
    df["automation_scaled"],
    c=df["automation_weighted_ratio"],
    cmap="RdYlGn_r",
    alpha=0.7,
    edgecolors="k",
    linewidths=0.3,
)
plt.colorbar(sc, label="automation_weighted_ratio")
max_val = max(df["augmentation_scaled"].max(), df["automation_scaled"].max())
ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1)
ax.set_xlabel("augmentation_weighted_ratio × pct_occ_scaled")
ax.set_ylabel("automation_weighted_ratio × pct_occ_scaled")
ax.set_title("Occupation Automation vs Augmentation (usage-weighted)")
plt.tight_layout()
plt.savefig("anthropic/scatter.png", dpi=150)
print("Saved to anthropic/scatter.png")

# Filtered version: clip to 95th percentile on both axes
p95_aug = df["augmentation_scaled"].quantile(0.95)
p95_aut = df["automation_scaled"].quantile(0.95)
df_filt = df[
    (df["augmentation_scaled"] <= p95_aug) & (df["automation_scaled"] <= p95_aut)
]

fig2, ax2 = plt.subplots(figsize=(10, 8))
sc2 = ax2.scatter(
    df_filt["augmentation_scaled"],
    df_filt["automation_scaled"],
    c=df_filt["automation_weighted_ratio"],
    cmap="RdYlGn_r",
    alpha=0.7,
    edgecolors="k",
    linewidths=0.3,
)
plt.colorbar(sc2, label="automation_weighted_ratio")
max_val2 = max(df_filt["augmentation_scaled"].max(), df_filt["automation_scaled"].max())
ax2.plot([0, max_val2], [0, max_val2], "k--", alpha=0.4, linewidth=1)
ax2.set_xlabel("augmentation_weighted_ratio × pct_occ_scaled")
ax2.set_ylabel("automation_weighted_ratio × pct_occ_scaled")
ax2.set_title("Occupation Automation vs Augmentation (usage-weighted, filtered)")
plt.tight_layout()
plt.savefig("anthropic/scatter_filtered.png", dpi=150)
print("Saved to anthropic/scatter_filtered.png")

# Augmentation-colored versions
fig3, ax3 = plt.subplots(figsize=(10, 8))
sc3 = ax3.scatter(
    df["augmentation_scaled"],
    df["automation_scaled"],
    c=df["augmentation_weighted_ratio"],
    cmap="RdYlGn_r",
    alpha=0.7,
    edgecolors="k",
    linewidths=0.3,
)
plt.colorbar(sc3, label="augmentation_weighted_ratio")
ax3.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1)
ax3.set_xlabel("augmentation_weighted_ratio × pct_occ_scaled")
ax3.set_ylabel("automation_weighted_ratio × pct_occ_scaled")
ax3.set_title(
    "Occupation Automation vs Augmentation (usage-weighted, colored by augmentation)"
)
plt.tight_layout()
plt.savefig("anthropic/scatter_aug.png", dpi=150)
print("Saved to anthropic/scatter_aug.png")

fig4, ax4 = plt.subplots(figsize=(10, 8))
sc4 = ax4.scatter(
    df_filt["augmentation_scaled"],
    df_filt["automation_scaled"],
    c=df_filt["augmentation_weighted_ratio"],
    cmap="RdYlGn_r",
    alpha=0.7,
    edgecolors="k",
    linewidths=0.3,
)
plt.colorbar(sc4, label="augmentation_weighted_ratio")
ax4.plot([0, max_val2], [0, max_val2], "k--", alpha=0.4, linewidth=1)
ax4.set_xlabel("augmentation_weighted_ratio × pct_occ_scaled")
ax4.set_ylabel("automation_weighted_ratio × pct_occ_scaled")
ax4.set_title(
    "Occupation Automation vs Augmentation"
    " (usage-weighted, filtered, colored by augmentation)"
)
plt.tight_layout()
plt.savefig("anthropic/scatter_aug_filtered.png", dpi=150)
print("Saved to anthropic/scatter_aug_filtered.png")
