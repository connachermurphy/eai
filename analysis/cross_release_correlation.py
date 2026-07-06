"""Cross-release and cross-platform correlation of AEI usage.

Builds two wide panels with task_count from each release × platform combination:
  - Task-level: O*NET task statements as the spine.
  - Occupation-level: aggregated to O*NET-SOC via equal-split apportionment
    (a task shared by N occupations contributes 1/N of its count to each).

For each panel, outputs:
  1. Release-vs-release (within platform): Jan vs Mar 2026 for API and claude.ai
     - 2×2 extensive margin confusion matrices (.md + .csv)
     - Scatter plots: untrimmed (left) + p95 trimmed (right)
  2. Platform-vs-platform (within release + pooled): API vs claude.ai
     - Same format as (1), for each release and pooled across releases
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eai.aei import RELEASES, load_aei_tasks
from eai.codebook import update_codebook
from eai.plot import apply_theme
from eai.utils import get_logger

log = get_logger(__name__)
apply_theme()
DATA = Path("output")
OUT = Path("output") / "cross_release"


# ==========================================================================
# Helpers
# ==========================================================================
def extensive_margin(a: pd.Series, b: pd.Series) -> dict:
    """Compute 2×2 confusion matrix for zero vs nonzero."""
    both_zero = int(((a == 0) & (b == 0)).sum())
    a_only = int(((a > 0) & (b == 0)).sum())
    b_only = int(((a == 0) & (b > 0)).sum())
    both_nonzero = int(((a > 0) & (b > 0)).sum())
    total = len(a)
    agreement = (both_zero + both_nonzero) / total
    return {
        "both_zero": both_zero,
        "a_only": a_only,
        "b_only": b_only,
        "both_nonzero": both_nonzero,
        "total": total,
        "agreement": agreement,
    }


def confusion_matrix_md(em: dict, label_a: str, label_b: str) -> str:
    """Format extensive margin as a 2×2 markdown confusion matrix with totals."""
    row_a0 = em["both_zero"] + em["b_only"]
    row_a1 = em["a_only"] + em["both_nonzero"]
    col_b0 = em["both_zero"] + em["a_only"]
    col_b1 = em["b_only"] + em["both_nonzero"]
    agreement_n = em["both_zero"] + em["both_nonzero"]
    lines = [
        f"| | {label_b} = 0 | {label_b} > 0 | **Total** |",
        "|---|---:|---:|---:|",
        (
            f"| **{label_a} = 0** | {em['both_zero']:,} | "
            f"{em['b_only']:,} | {row_a0:,} |"
        ),
        (
            f"| **{label_a} > 0** | {em['a_only']:,} | "
            f"{em['both_nonzero']:,} | {row_a1:,} |"
        ),
        f"| **Total** | {col_b0:,} | {col_b1:,} | {em['total']:,} |",
        "",
        f"Agreement: {em['agreement']:.1%} ({agreement_n:,} / {em['total']:,})",
    ]
    return "\n".join(lines)


def compute_correlations(a: pd.Series, b: pd.Series) -> dict:
    """Pearson and Spearman correlations: all units, both-nonzero, p95-trimmed."""
    mask = (a > 0) & (b > 0)
    sa, sb = a[mask], b[mask]
    n_bnz = len(sa)

    if n_bnz > 1:
        p95_a = sa.quantile(0.95)
        p95_b = sb.quantile(0.95)
        trim = (sa <= p95_a) & (sb <= p95_b)
        ta, tb = sa[trim], sb[trim]
    else:
        ta, tb = sa, sb
    n_trim = len(ta)

    def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
        return x.corr(y, method=method) if len(x) > 1 else float("nan")

    return {
        "n_both_nonzero": n_bnz,
        "n_trimmed": n_trim,
        "pearson_all": _corr(a, b, "pearson"),
        "spearman_all": _corr(a, b, "spearman"),
        "pearson_both_nonzero": _corr(sa, sb, "pearson"),
        "spearman_both_nonzero": _corr(sa, sb, "spearman"),
        "pearson_trimmed": _corr(ta, tb, "pearson"),
        "spearman_trimmed": _corr(ta, tb, "spearman"),
    }


def correlations_md(corrs: dict) -> str:
    """Format correlation table (Pearson + Spearman across subsets)."""

    def fmt(x: float) -> str:
        return f"{x:.3f}" if pd.notna(x) else "—"

    return "\n".join(
        [
            "| Subset | Pearson | Spearman | n |",
            "|---|---:|---:|---:|",
            (
                f"| All units | {fmt(corrs['pearson_all'])} | "
                f"{fmt(corrs['spearman_all'])} | — |"
            ),
            (
                f"| Both nonzero | {fmt(corrs['pearson_both_nonzero'])} | "
                f"{fmt(corrs['spearman_both_nonzero'])} | "
                f"{corrs['n_both_nonzero']:,} |"
            ),
            (
                "| Both nonzero, p95 trimmed | "
                f"{fmt(corrs['pearson_trimmed'])} | "
                f"{fmt(corrs['spearman_trimmed'])} | "
                f"{corrs['n_trimmed']:,} |"
            ),
        ]
    )


def scatter_rank(
    a: pd.Series,
    b: pd.Series,
    label_a: str,
    label_b: str,
    title: str,
    path: Path,
    spearman: float,
) -> None:
    """Rank-rank scatter on the both-nonzero subset (Spearman ρ in title)."""
    mask = (a > 0) & (b > 0)
    sa, sb = a[mask], b[mask]
    n = len(sa)
    if n < 2:
        return

    rank_a = sa.rank()
    rank_b = sb.rank()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(rank_a, rank_b, alpha=0.3, s=10, edgecolors="none")
    ax.plot([1, n], [1, n], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel(f"{label_a} rank")
    ax.set_ylabel(f"{label_b} rank")
    ax.set_title(f"{title}\nrank-rank (n={n:,}, rho={spearman:.3f})")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def scatter_pair(
    a: pd.Series,
    b: pd.Series,
    label_a: str,
    label_b: str,
    title: str,
    path: Path,
) -> None:
    """Two-panel scatter: both nonzero untrimmed (left) + p95 trimmed (right).

    Also saves a standalone trimmed-only figure at {stem}_trimmed.png.
    """
    mask = (a > 0) & (b > 0)
    sa, sb = a[mask], b[mask]
    n = len(sa)
    corr = sa.corr(sb) if n > 1 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: both nonzero, untrimmed
    ax = axes[0]
    ax.scatter(sa, sb, alpha=0.3, s=10, edgecolors="none")
    if n > 0:
        max_val = max(sa.max(), sb.max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.set_title(f"Both nonzero (n={n:,}, r={corr:.3f})")

    # Right: p95 trimmed
    ax = axes[1]
    if n > 1:
        p95_a = sa.quantile(0.95)
        p95_b = sb.quantile(0.95)
        trim = (sa <= p95_a) & (sb <= p95_b)
        ta, tb = sa[trim], sb[trim]
        corr_trim = ta.corr(tb) if len(ta) > 1 else float("nan")
        ax.scatter(ta, tb, alpha=0.3, s=10, edgecolors="none")
        if len(ta) > 0:
            max_val = max(ta.max(), tb.max())
            ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1)
        ax.set_title(f"p95 trimmed (n={len(ta):,}, r={corr_trim:.3f})")
    else:
        ax.set_title("p95 trimmed (insufficient data)")
        ta, tb, corr_trim = sa, sb, corr
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)

    # Standalone trimmed figure
    if n > 1:
        fig_t, ax_t = plt.subplots(figsize=(6, 5))
        ax_t.scatter(ta, tb, alpha=0.3, s=10, edgecolors="none")
        if len(ta) > 0:
            max_val = max(ta.max(), tb.max())
            ax_t.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1)
        ax_t.set_xlabel(label_a)
        ax_t.set_ylabel(label_b)
        ax_t.set_title(f"{title}\np95 trimmed (n={len(ta):,}, r={corr_trim:.3f})")
        plt.tight_layout()
        trimmed_path = path.with_name(f"{path.stem}_trimmed{path.suffix}")
        plt.savefig(trimmed_path, dpi=150)
        plt.close(fig_t)


def save_comparison(
    a: pd.Series,
    b: pd.Series,
    label_a: str,
    label_b: str,
    name: str,
    fig_dir: Path,
    table_dir: Path,
) -> None:
    """Save extensive margin + correlation tables (.md + .csv) and scatter plots."""
    em = extensive_margin(a, b)
    corrs = compute_correlations(a, b)

    # Markdown
    md = (
        f"## {label_a} vs {label_b}\n\n"
        f"{confusion_matrix_md(em, label_a, label_b)}\n\n"
        f"**Correlations:**\n\n"
        f"{correlations_md(corrs)}\n"
    )
    md_path = table_dir / f"{name}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)

    # CSV
    csv_df = pd.DataFrame([em | corrs | {"label_a": label_a, "label_b": label_b}])
    csv_df.to_csv(table_dir / f"{name}.csv", index=False)

    # Scatters
    title = f"{label_a} vs {label_b}"
    scatter_pair(a, b, label_a, label_b, title, fig_dir / f"{name}.png")
    scatter_rank(
        a,
        b,
        label_a,
        label_b,
        title,
        fig_dir / f"{name}_rank.png",
        corrs["spearman_both_nonzero"],
    )

    log.info(
        "%s: agreement=%.1f%%, n_bnz=%d, pearson=%.3f, spearman=%.3f",
        name,
        em["agreement"] * 100,
        corrs["n_both_nonzero"],
        corrs["pearson_both_nonzero"],
        corrs["spearman_both_nonzero"],
    )


# ==========================================================================
# Short labels
# ==========================================================================
RELEASE_LABELS = {
    "release_2025_09_15": "2025_09",
    "release_2026_01_15": "2026_01",
    "release_2026_03_24": "2026_03",
}


# ==========================================================================
# Comparisons (run on either task or occupation panel)
# ==========================================================================
def run_comparisons(panel: pd.DataFrame, level: str) -> None:
    """Run release- and platform-pair comparisons on a panel.

    `level` controls the output subdirectory (`task` or `occupation`).
    """
    fig_dir = OUT / level / "figures"
    table_dir = OUT / level / "tables"

    log.info("=== %s: Release-vs-release comparisons ===", level)
    save_comparison(
        panel["task_count_1p_api_2026_01"],
        panel["task_count_1p_api_2026_03"],
        "API Jan 2026",
        "API Mar 2026",
        "release_jan_vs_mar_2026_api",
        fig_dir,
        table_dir,
    )
    save_comparison(
        panel["task_count_claude_ai_2026_01"],
        panel["task_count_claude_ai_2026_03"],
        "Claude.ai Jan 2026",
        "Claude.ai Mar 2026",
        "release_jan_vs_mar_2026_claude_ai",
        fig_dir,
        table_dir,
    )

    log.info("=== %s: Platform-vs-platform comparisons ===", level)
    for _release, label in RELEASE_LABELS.items():
        save_comparison(
            panel[f"task_count_1p_api_{label}"],
            panel[f"task_count_claude_ai_{label}"],
            f"API {label}",
            f"Claude.ai {label}",
            f"platform_api_vs_claude_{label}",
            fig_dir,
            table_dir,
        )

    save_comparison(
        panel["task_count_1p_api_pooled"],
        panel["task_count_claude_ai_pooled"],
        "API (pooled)",
        "Claude.ai (pooled)",
        "platform_api_vs_claude_pooled",
        fig_dir,
        table_dir,
    )

    log.info("Tables: %s", table_dir)
    log.info("Figures: %s", fig_dir)


# ==========================================================================
# Build task-level panel
# ==========================================================================
onet = pd.read_csv(DATA / "onet" / "task_statements.csv")
onet["task_key"] = onet["Task"].str.lower().str.strip()
tasks = onet[["task_key"]].drop_duplicates().reset_index(drop=True)
log.info("O*NET unique tasks: %d", len(tasks))

aei = load_aei_tasks(logger=log)

for release in RELEASES:
    for platform in RELEASES[release]:
        mask = (aei["release"] == release) & (aei["platform"] == platform)
        subset = aei.loc[mask, ["task_key", "task_count"]]

        n_dupes = subset["task_key"].duplicated().sum()
        if n_dupes > 0:
            log.warning(
                "  %s / %s: %d duplicate task keys — taking first",
                release,
                platform,
                n_dupes,
            )
            subset = subset.drop_duplicates(subset="task_key", keep="first")

        suffix = f"{platform}_{RELEASE_LABELS[release]}"
        subset = subset.rename(columns={"task_count": f"task_count_{suffix}"})

        tasks = tasks.merge(subset, on="task_key", how="left")
        n_matched = tasks[f"task_count_{suffix}"].notna().sum()
        log.info("  %s / %s: %d tasks matched", release, platform, n_matched)

# Fill unobserved tasks with zero
count_cols = [c for c in tasks.columns if c != "task_key"]
tasks[count_cols] = tasks[count_cols].fillna(0)

# Pooled: sum task_count across releases for each platform
api_cols = [f"task_count_1p_api_{label}" for label in RELEASE_LABELS.values()]
claude_cols = [f"task_count_claude_ai_{label}" for label in RELEASE_LABELS.values()]
tasks["task_count_1p_api_pooled"] = tasks[api_cols].sum(axis=1)
tasks["task_count_claude_ai_pooled"] = tasks[claude_cols].sum(axis=1)

log.info("Task panel shape: %d tasks × %d columns", len(tasks), len(tasks.columns))
tasks.to_csv(DATA / "cross_release_panel_task.csv", index=False)
log.info("Saved task panel: %s", DATA / "cross_release_panel_task.csv")


# ==========================================================================
# Build occupation-level panel (equal-split apportionment)
# ==========================================================================
def build_occupation_panel(
    task_panel: pd.DataFrame, onet: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate task panel to O*NET-SOC via equal-split apportionment.

    A task that appears in N occupations contributes 1/N of its count to each.
    Apportionment weights for each task sum to 1, so total counts are preserved
    across the aggregation.
    """
    mapping = onet[["task_key", "O*NET-SOC Code"]].drop_duplicates()
    mapping["weight"] = 1 / mapping.groupby("task_key")["O*NET-SOC Code"].transform(
        "count"
    )

    cols = [c for c in task_panel.columns if c != "task_key"]
    long = task_panel.merge(mapping, on="task_key", how="inner")
    long[cols] = long[cols].mul(long["weight"], axis=0)

    return long.groupby("O*NET-SOC Code", as_index=False)[cols].sum()


occupations = build_occupation_panel(tasks, onet)
log.info(
    "Occupation panel shape: %d occupations × %d columns",
    len(occupations),
    len(occupations.columns),
)
occupations.to_csv(DATA / "cross_release_panel_occupation.csv", index=False)
log.info("Saved occupation panel: %s", DATA / "cross_release_panel_occupation.csv")


# ==========================================================================
# Run comparisons at both aggregation levels
# ==========================================================================
run_comparisons(tasks, "task")
run_comparisons(occupations, "occupation")

# ==========================================================================
# Update codebooks
# ==========================================================================
COUNT_PATTERN_COLS = [
    (
        "task_count_{platform}_{release}",
        "Raw AEI conversation count for the unit in the given platform "
        "(1p_api or claude_ai) and release (2025_09, 2026_01, or 2026_03). "
        "Units unobserved in a release are filled with 0.",
    ),
    (
        "task_count_{platform}_pooled",
        "Sum of the platform's counts across the three releases.",
    ),
]
update_codebook(
    DATA / "codebook.md",
    section="cross_release_panels",
    title="Cross-release panels",
    source="analysis/cross_release_correlation.py",
    files=[
        {
            "name": "cross_release_panel_task.csv",
            "description": (
                "Wide task-level panel of AEI conversation counts, one row "
                "per O*NET task, one column per platform-release."
            ),
            "columns": [
                (
                    "task_key",
                    "O*NET task statement text, lowercased and stripped; the "
                    "key used to match tasks across releases.",
                )
            ]
            + COUNT_PATTERN_COLS,
        },
        {
            "name": "cross_release_panel_occupation.csv",
            "description": (
                "The task panel aggregated to O*NET-SOC occupations by "
                "equal-split apportionment: a task in N occupations "
                "contributes 1/N of its count to each."
            ),
            "columns": [
                ("O*NET-SOC Code", "Eight-character O*NET-SOC occupation code.")
            ]
            + COUNT_PATTERN_COLS,
        },
    ],
)
update_codebook(
    OUT / "codebook.md",
    section="cross_release_tables",
    title="Cross-release comparison tables",
    source="analysis/cross_release_correlation.py",
    intro=(
        "The task/ and occupation/ subdirectories hold the same comparisons "
        "at the two aggregation levels. Each comparison writes a one-row CSV "
        "(columns below), a markdown rendering, and scatter figures."
    ),
    files=[
        {
            "name": "{task,occupation}/tables/*.csv",
            "description": (
                "One row per comparison of two count series (release vs "
                "release or platform vs platform)."
            ),
            "columns": [
                (
                    "both_zero / a_only / b_only / both_nonzero",
                    "Extensive-margin cell counts: units with zero counts in "
                    "both series, only in series A, only in series B, or "
                    "nonzero in both.",
                ),
                ("total", "Total units compared."),
                (
                    "agreement",
                    "(both_zero + both_nonzero) / total: share of units where "
                    "the two series agree on observed vs unobserved.",
                ),
                ("n_both_nonzero", "Units in the both-nonzero subset."),
                (
                    "n_trimmed",
                    "Units in the both-nonzero subset after trimming values "
                    "above either series' 95th percentile.",
                ),
                (
                    "pearson_all / spearman_all",
                    "Correlations over all units, including zeros.",
                ),
                (
                    "pearson_both_nonzero / spearman_both_nonzero",
                    "Correlations over the both-nonzero subset.",
                ),
                (
                    "pearson_trimmed / spearman_trimmed",
                    "Correlations over the p95-trimmed both-nonzero subset.",
                ),
                ("label_a / label_b", "Display labels of the two series."),
            ],
        }
    ],
)
log.info("Updated codebooks: %s, %s", DATA / "codebook.md", OUT / "codebook.md")
