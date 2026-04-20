"""Cross-release correlation of AEI task-level usage.

Builds a single wide DataFrame with O*NET task statements as the spine,
merging task_count, automation_count, and augmentation_count from each
release × platform combination. Tasks not observed in a release are coded
as zero.

Analyses:
  1. Overall correlation matrices (all tasks, including zeros)
  2. Extensive margin: confusion matrices (zero vs nonzero agreement)
  3. Intensive margin (either nonzero): correlation among tasks with usage in
     at least one of the pair
  4. Intensive margin (both nonzero): correlation among tasks with usage in both

Outputs:
  - data/cross_release_panel.csv: the wide panel
  - analysis/figures/scatter_{measure}_{a}_vs_{b}.png: pairwise scatters
"""

import itertools
import logging
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
ONET_CSV = BASE / "data" / "onet" / "task_statements.csv"
DATA = BASE / "data"
OUT_CSV = DATA / "cross_release_panel.csv"

# Release registry (must match anthropic/clean.py)
RELEASES = {
    "release_2025_09_15": ["claude_ai", "1p_api"],
    "release_2026_01_15": ["claude_ai", "1p_api"],
    "release_2026_03_24": ["claude_ai", "1p_api"],
}

# Short labels for column suffixes
RELEASE_LABELS = {
    "release_2025_09_15": "2025_09",
    "release_2026_01_15": "2026_01",
    "release_2026_03_24": "2026_03",
}

AUTOMATION_COLS = ["directive_count", "feedback_loop_count"]
AUGMENTATION_COLS = ["validation_count", "task_iteration_count", "learning_count"]


def load_and_summarize(release: str, platform: str) -> pd.DataFrame:
    """Load a cleaned AEI file and compute automation/augmentation counts."""
    path = DATA / release / f"aei_cleaned_{platform}.csv"
    df = pd.read_csv(path)

    # Compute derived counts
    df["automation_count"] = df[AUTOMATION_COLS].sum(axis=1)
    df["augmentation_count"] = df[AUGMENTATION_COLS].sum(axis=1)

    # Build merge key
    df["task_merge_key"] = df["task"].str.lower().str.strip()

    suffix = f"{platform}_{RELEASE_LABELS[release]}"
    return df[["task_merge_key", "task_count", "automation_count", "augmentation_count"]].rename(
        columns={
            "task_count": f"task_count_{suffix}",
            "automation_count": f"automation_count_{suffix}",
            "augmentation_count": f"augmentation_count_{suffix}",
        }
    )


# ==========================================================================
# Build panel
# ==========================================================================
onet = pd.read_csv(ONET_CSV)
onet["task_merge_key"] = onet["Task"].str.lower().str.strip()

# Deduplicate O*NET tasks (same task text can appear under multiple occupations)
tasks = onet[["task_merge_key"]].drop_duplicates().reset_index(drop=True)
logger.info("O*NET unique tasks: %d", len(tasks))

for release, platforms in RELEASES.items():
    for platform in platforms:
        summary = load_and_summarize(release, platform)

        # Check for duplicate keys in summary (shouldn't happen in cleaned data)
        n_dupes = summary["task_merge_key"].duplicated().sum()
        if n_dupes > 0:
            logger.warning(
                "  %s / %s: %d duplicate task keys — taking first",
                release,
                platform,
                n_dupes,
            )
            summary = summary.drop_duplicates(subset="task_merge_key", keep="first")

        tasks = tasks.merge(summary, on="task_merge_key", how="left")
        suffix = f"{platform}_{RELEASE_LABELS[release]}"
        n_matched = tasks[f"task_count_{suffix}"].notna().sum()
        logger.info(
            "  %s / %s: %d tasks matched", release, platform, n_matched
        )

# Fill unobserved tasks with zero
count_cols = [c for c in tasks.columns if c != "task_merge_key"]
tasks[count_cols] = tasks[count_cols].fillna(0)

logger.info("Panel shape: %d tasks × %d columns", len(tasks), len(tasks.columns))

# ==========================================================================
# Analysis
# ==========================================================================
measures = ["task_count", "automation_count", "augmentation_count"]
FIG_DIR = BASE / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def short_name(col: str, measure: str) -> str:
    """Strip measure prefix from column name for display."""
    return col.replace(f"{measure}_", "")


TABLE_DIR = BASE / "analysis" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
all_extensive = []
all_summary = []

for measure in measures:
    cols = [c for c in tasks.columns if c.startswith(f"{measure}_")]
    short_names = [short_name(c, measure) for c in cols]

    # --- Pairwise analyses (within-platform only) ---
    logger.info("--- %s: pairwise analyses ---", measure)
    all_pairs = list(itertools.combinations(range(len(cols)), 2))
    # Keep only pairs where both columns share the same platform prefix
    pairs = [
        (i, j)
        for i, j in all_pairs
        if short_names[i].rsplit("_", 2)[0] == short_names[j].rsplit("_", 2)[0]
    ]

    # Collect results
    extensive_rows = []
    summary_rows = []

    for i, j in pairs:
        col_a, col_b = cols[i], cols[j]
        name_a, name_b = short_names[i], short_names[j]
        pair_label = f"{name_a} vs {name_b}"
        a, b = tasks[col_a], tasks[col_b]

        # --- Extensive margin: confusion matrix ---
        both_zero = ((a == 0) & (b == 0)).sum()
        a_only = ((a > 0) & (b == 0)).sum()
        b_only = ((a == 0) & (b > 0)).sum()
        both_nonzero = ((a > 0) & (b > 0)).sum()
        total = len(a)
        agreement = (both_zero + both_nonzero) / total

        extensive_rows.append(
            {
                "measure": measure,
                "pair": pair_label,
                "both_zero": both_zero,
                "a_only": a_only,
                "b_only": b_only,
                "both_nonzero": both_nonzero,
                "agreement": agreement,
            }
        )

        # --- Correlations across filters ---
        mask_either = (a > 0) | (b > 0)
        mask_both = (a > 0) & (b > 0)

        row = {
            "measure": measure,
            "pair": pair_label,
            # All tasks (including zeros)
            "n_all": total,
            "corr_all": a.corr(b),
        }

        for mask, label in [
            (mask_either, "either_nonzero"),
            (mask_both, "both_nonzero"),
        ]:
            n = mask.sum()
            sa, sb = a[mask], b[mask]
            corr_full = sa.corr(sb) if n > 1 else float("nan")
            row[f"n_{label}"] = n
            row[f"corr_{label}"] = corr_full

            # p95 trimmed
            if n > 1:
                p95_a = sa.quantile(0.95)
                p95_b = sb.quantile(0.95)
                trim = (sa <= p95_a) & (sb <= p95_b)
                ta, tb = sa[trim], sb[trim]
                corr_p95 = ta.corr(tb) if len(ta) > 1 else float("nan")
                row[f"n_{label}_p95"] = len(ta)
                row[f"corr_{label}_p95"] = corr_p95
            else:
                row[f"n_{label}_p95"] = 0
                row[f"corr_{label}_p95"] = float("nan")

        summary_rows.append(row)

        # --- Scatter plots ---
        n_either = mask_either.sum()
        n_both = mask_both.sum()
        corr_either = row["corr_either_nonzero"]
        corr_both = row["corr_both_nonzero"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Row 1: full range
        ax = axes[0, 0]
        ax.scatter(
            a[mask_either], b[mask_either], alpha=0.3, s=10, edgecolors="none"
        )
        max_val = max(a[mask_either].max(), b[mask_either].max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel(name_a)
        ax.set_ylabel(name_b)
        ax.set_title(f"Either nonzero (n={n_either:,}, r={corr_either:.3f})")

        ax = axes[0, 1]
        if n_both > 0:
            ax.scatter(
                a[mask_both], b[mask_both], alpha=0.3, s=10, edgecolors="none"
            )
            max_val = max(a[mask_both].max(), b[mask_both].max())
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel(name_a)
        ax.set_ylabel(name_b)
        ax.set_title(f"Both nonzero (n={n_both:,}, r={corr_both:.3f})")

        # Row 2: trimmed at 95th percentile
        for col_idx, (mask, label) in enumerate(
            [(mask_either, "Either nonzero"), (mask_both, "Both nonzero")]
        ):
            ax = axes[1, col_idx]
            subset_a, subset_b = a[mask], b[mask]
            if len(subset_a) < 2:
                ax.set_title(f"{label}, p95 (insufficient data)")
                continue
            p95_a = subset_a.quantile(0.95)
            p95_b = subset_b.quantile(0.95)
            trim = (subset_a <= p95_a) & (subset_b <= p95_b)
            ta, tb = subset_a[trim], subset_b[trim]
            corr_trim = ta.corr(tb) if len(ta) > 1 else float("nan")
            ax.scatter(ta, tb, alpha=0.3, s=10, edgecolors="none")
            if len(ta) > 0:
                max_val = max(ta.max(), tb.max())
                ax.plot(
                    [0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1
                )
            ax.set_xlabel(name_a)
            ax.set_ylabel(name_b)
            ax.set_title(
                f"{label}, p95 trim (n={len(ta):,}, r={corr_trim:.3f})"
            )

        fig.suptitle(f"{measure}: {name_a} vs {name_b}", fontsize=11)
        plt.tight_layout()
        fig_path = FIG_DIR / f"scatter_{measure}_{name_a}_vs_{name_b}.png"
        plt.savefig(fig_path, dpi=100)
        plt.close(fig)

    # Log summary tables
    ext_df = pd.DataFrame(extensive_rows)
    logger.info(
        "\n  Extensive margin:\n%s",
        ext_df.to_string(index=False, float_format="%.4f"),
    )

    summ_df = pd.DataFrame(summary_rows)
    logger.info(
        "\n  Correlations:\n%s",
        summ_df.to_string(index=False, float_format="%.4f"),
    )

    all_extensive.append(ext_df)
    all_summary.append(summ_df)

# ==========================================================================
# Write
# ==========================================================================
tasks.to_csv(OUT_CSV, index=False)
logger.info("Wrote panel to %s", OUT_CSV)

extensive_df = pd.concat(all_extensive, ignore_index=True)
extensive_df.to_csv(TABLE_DIR / "extensive_margin.csv", index=False)

summary_df = pd.concat(all_summary, ignore_index=True)
summary_df.to_csv(TABLE_DIR / "correlations.csv", index=False)

logger.info("Wrote tables to %s", TABLE_DIR)
logger.info("Figures saved to %s", FIG_DIR)
