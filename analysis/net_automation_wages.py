"""Plot AEI net automation usage against OEWS wages.

Net automation usage is automation usage minus augmentation usage. The input
panel is expected to be produced by occupational_characteristics.py.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from eai.plot import apply_theme
from eai.utils import get_logger

log = get_logger(__name__)
apply_theme()

DEFAULT_PANEL = Path("output") / "occupations_aei_oews_2024.csv"
DEFAULT_OUT = Path("output") / "net_automation_wages"
WAGE_COL = "oews_a_mean"
EMP_COL = "oews_tot_emp_imputed"
NET_COL_PREFIX = "net_emp_pc"
DEFAULT_WINSOR_LOWER = 0.01
DEFAULT_WINSOR_UPPER = 0.99
NOTE = (
    "Note: shared O*NET task usage is apportioned across linked occupations in "
    "proportion to 2024 OEWS employment, then divided by imputed occupation "
    "employment. Net automation usage is winsorized at p1/p99."
)

PLATFORMS = {
    "1p_api": "API",
    "claude_ai": "Claude.ai",
    "total": "Total",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Plot AEI net automation usage against OEWS wages."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_PANEL,
        help=f"Occupation-level AEI/OEWS panel (default: {DEFAULT_PANEL}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--winsor-lower",
        type=float,
        default=DEFAULT_WINSOR_LOWER,
        help=f"Lower winsorization quantile (default: {DEFAULT_WINSOR_LOWER}).",
    )
    parser.add_argument(
        "--winsor-upper",
        type=float,
        default=DEFAULT_WINSOR_UPPER,
        help=f"Upper winsorization quantile (default: {DEFAULT_WINSOR_UPPER}).",
    )
    return parser.parse_args()


def net_col(platform: str) -> str:
    """Return the derived net automation column name."""
    return f"{NET_COL_PREFIX}_{platform}_count_pc"


def winsor_col(platform: str) -> str:
    """Return the winsorized net automation column name."""
    return f"{net_col(platform)}_winsorized"


def source_col(platform: str, measure: str) -> str:
    """Return the source automation/augmentation column name."""
    return f"emp_{platform}_{measure}_count_pc"


def validate_winsor_bounds(lower: float, upper: float) -> None:
    """Validate winsorization quantile bounds."""
    if not 0 <= lower < upper <= 1:
        raise ValueError(
            "Expected winsorization bounds to satisfy "
            f"0 <= lower < upper <= 1, got lower={lower}, upper={upper}"
        )


def add_net_columns(
    df: pd.DataFrame,
    winsor_lower: float,
    winsor_upper: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Add raw and winsorized per-capita net automation columns."""
    df = df.copy()
    winsor_bounds = {}
    for platform in PLATFORMS:
        automation = source_col(platform, "automation")
        augmentation = source_col(platform, "augmentation")
        raw_col = net_col(platform)
        clipped_col = winsor_col(platform)
        df[raw_col] = df[automation] - df[augmentation]

        lower = df[raw_col].quantile(winsor_lower)
        upper = df[raw_col].quantile(winsor_upper)
        df[clipped_col] = df[raw_col].clip(lower=lower, upper=upper)
        winsor_bounds[platform] = {
            "winsor_lower_quantile": winsor_lower,
            "winsor_upper_quantile": winsor_upper,
            "winsor_lower_value": lower,
            "winsor_upper_value": upper,
        }
    return df, winsor_bounds


def correlation_rows(
    df: pd.DataFrame,
    winsor_bounds: dict[str, dict[str, float]],
    nonzero_only: bool = False,
) -> list[dict]:
    """Compute wage/net automation correlations for each derived column."""
    rows = []
    for platform, platform_label in PLATFORMS.items():
        raw_col = net_col(platform)
        col = winsor_col(platform)
        data = df[[WAGE_COL, raw_col, col]].dropna()
        if nonzero_only:
            data = data[data[raw_col] != 0]
        rows.append(
            {
                "platform": platform,
                "platform_label": platform_label,
                "net_column": col,
                "nonzero_only": nonzero_only,
                **winsor_bounds[platform],
                "n": len(data),
                "pearson": data[WAGE_COL].corr(data[col], method="pearson"),
                "spearman": data[WAGE_COL].corr(data[col], method="spearman"),
                "mean_net": data[col].mean(),
                "median_net": data[col].median(),
                "n_positive": int((data[col] > 0).sum()),
                "n_negative": int((data[col] < 0).sum()),
                "n_zero": int((data[col] == 0).sum()),
            }
        )
    return rows


def format_wage(x: float, _pos: int) -> str:
    """Format wage ticks as thousands of dollars."""
    return f"${x / 1000:.0f}k"


def format_per_capita(x: float, _pos: int) -> str:
    """Format per-capita net usage ticks compactly."""
    return f"{x:.3g}"


def save_scatter_grid(
    df: pd.DataFrame,
    out_dir: Path,
    nonzero_only: bool = False,
) -> Path:
    """Save one 3-panel wage scatter grid."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True)

    for ax, (platform, platform_label) in zip(axes, PLATFORMS.items(), strict=True):
        raw_col = net_col(platform)
        col = winsor_col(platform)
        plot_df = df[[WAGE_COL, EMP_COL, raw_col, col]].dropna()
        if nonzero_only:
            plot_df = plot_df[plot_df[raw_col] != 0]
        sizes = 12 + 30 * plot_df[EMP_COL].rank(pct=True)

        ax.scatter(
            plot_df[col],
            plot_df[WAGE_COL],
            s=sizes,
            alpha=0.55,
            linewidths=0,
        )
        ax.axvline(0, color="black", linewidth=1, alpha=0.55)

        pearson = plot_df[WAGE_COL].corr(plot_df[col], method="pearson")
        spearman = plot_df[WAGE_COL].corr(plot_df[col], method="spearman")
        ax.set_title(
            f"{platform_label}\nr={pearson:.2f}, rho={spearman:.2f}, n={len(plot_df):,}"
        )
        ax.set_xlabel("Automation minus augmentation usage per worker")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_per_capita))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_wage))

    axes[0].set_ylabel("2024 OEWS annual mean wage")
    title = "Net automation usage and wages by platform"
    if nonzero_only:
        title += " (nonzero usage)"
    fig.suptitle(title, fontsize=13)
    fig.text(0.01, 0.02, NOTE, ha="left", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_nonzero" if nonzero_only else ""
    path = out_dir / f"net_automation_vs_wages{suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    """Run the analysis."""
    args = parse_args()
    validate_winsor_bounds(args.winsor_lower, args.winsor_upper)
    if not args.panel.exists():
        raise FileNotFoundError(f"Missing input panel: {args.panel}")

    df = pd.read_csv(args.panel)
    log.info("Loaded %s: %d occupations", args.panel, len(df))

    df, winsor_bounds = add_net_columns(
        df,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
    )

    net_columns = [
        col
        for platform in PLATFORMS
        for col in [net_col(platform), winsor_col(platform)]
    ]
    analysis_cols = ["soc_2010", "title_2010", "group_id", EMP_COL, WAGE_COL]
    net_path = args.out / "occupation_net_automation_usage.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    df[analysis_cols + net_columns].to_csv(net_path, index=False)
    log.info("Saved net automation panel: %s", net_path)

    corr = pd.DataFrame(
        correlation_rows(df, winsor_bounds)
        + correlation_rows(df, winsor_bounds, nonzero_only=True)
    )
    corr_path = args.out / "net_automation_wage_correlations.csv"
    corr.to_csv(corr_path, index=False)
    log.info("Saved correlations: %s", corr_path)

    path = save_scatter_grid(df, args.out / "figures")
    log.info("Saved figure: %s", path)
    path = save_scatter_grid(df, args.out / "figures", nonzero_only=True)
    log.info("Saved figure: %s", path)


if __name__ == "__main__":
    main()
