"""Plot AEI net automation usage against OEWS wages.

Net automation usage is automation usage minus augmentation usage. The input
panel is expected to be produced by occupational_characteristics.py.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from eai.codebook import update_codebook
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
POINT_COLOR = "#4C72B0"
POINT_SIZE_MIN = 14
POINT_SIZE_MAX = 80
POINT_SIZE_LOWER = 0.05
POINT_SIZE_UPPER = 0.95
SOURCES = (
    "Sources: Anthropic Economic Index (Massenkoff et al. 2026); "
    "BLS OEWS May 2024 national estimates."
)
FIGURE_NOTE = (
    "Winsorized p1/p99. Marker size: imputed employment, clipped p5/p95. "
    "Gray line: weighted OLS. $r_w$ and $\\rho_w$ are employment-weighted; "
    "parentheses show unweighted $\\rho$ and n."
)
METHOD_NOTE = (
    "Shared task usage is apportioned by 2024 OEWS employment, then divided by "
    "imputed occupation employment."
)
NOTE = f"{METHOD_NOTE} {FIGURE_NOTE}"
NOTE_TEXT = (
    f"Notes: {NOTE}".replace("$\\rho_w$", "ρw")
    .replace("$r_w$", "rw")
    .replace("$\\rho$", "ρ")
)
SOURCE_DETAILS = "\n".join(
    [
        "Sources:",
        (
            "- Anthropic Economic Index: Massenkoff, Maxim; Eva Lyubich; "
            "Peter McCrory; Ruth Appel; and Ryan Heller. 2026. "
            '"Anthropic Economic Index report: Learning curves." '
            "March 24, 2026. "
            "https://www.anthropic.com/research/economic-index-march-2026-report"
        ),
        (
            "- Bureau of Labor Statistics Occupational Employment and Wage "
            "Statistics: May 2024 national occupational employment and wage "
            "estimates. https://www.bls.gov/oes/tables.htm"
        ),
    ]
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


def weighted_corr(x: pd.Series, y: pd.Series, weights: pd.Series) -> float:
    """Compute an employment-weighted Pearson correlation."""
    data = pd.concat([x, y, weights], axis=1).dropna()
    if len(data) < 2:
        return np.nan

    x_values = data.iloc[:, 0].astype(float)
    y_values = data.iloc[:, 1].astype(float)
    w_values = data.iloc[:, 2].astype(float)
    positive = w_values > 0
    x_values = x_values[positive]
    y_values = y_values[positive]
    w_values = w_values[positive]
    if len(x_values) < 2 or w_values.sum() <= 0:
        return np.nan

    x_mean = np.average(x_values, weights=w_values)
    y_mean = np.average(y_values, weights=w_values)
    x_centered = x_values - x_mean
    y_centered = y_values - y_mean
    cov = np.average(x_centered * y_centered, weights=w_values)
    x_var = np.average(x_centered**2, weights=w_values)
    y_var = np.average(y_centered**2, weights=w_values)
    if x_var <= 0 or y_var <= 0:
        return np.nan
    return cov / np.sqrt(x_var * y_var)


def weighted_rank_corr(x: pd.Series, y: pd.Series, weights: pd.Series) -> float:
    """Compute an employment-weighted Spearman rank correlation."""
    return weighted_corr(x.rank(), y.rank(), weights)


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
        data = df[[WAGE_COL, EMP_COL, raw_col, col]].dropna()
        if nonzero_only:
            data = data[data[raw_col] != 0]
        data = data[data[EMP_COL] > 0]
        weighted_pearson = weighted_corr(data[WAGE_COL], data[col], data[EMP_COL])
        weighted_spearman = weighted_rank_corr(
            data[WAGE_COL],
            data[col],
            data[EMP_COL],
        )
        rows.append(
            {
                "platform": platform,
                "platform_label": platform_label,
                "net_column": col,
                "nonzero_only": nonzero_only,
                "weight_column": EMP_COL,
                **winsor_bounds[platform],
                "n": len(data),
                "pearson": weighted_pearson,
                "spearman": weighted_spearman,
                "pearson_unweighted": data[WAGE_COL].corr(
                    data[col],
                    method="pearson",
                ),
                "spearman_unweighted": data[WAGE_COL].corr(
                    data[col],
                    method="spearman",
                ),
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


def add_best_fit_line(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    weights: pd.Series,
) -> None:
    """Draw a light employment-weighted OLS line for y on x."""
    if len(x) < 2 or x.nunique() < 2:
        return

    data = pd.concat([x, y, weights], axis=1).dropna()
    data = data[data.iloc[:, 2] > 0]
    if len(data) < 2:
        return

    x_values = data.iloc[:, 0].astype(float)
    y_values = data.iloc[:, 1].astype(float)
    w_values = data.iloc[:, 2].astype(float)
    x_mean = np.average(x_values, weights=w_values)
    y_mean = np.average(y_values, weights=w_values)
    x_centered = x_values - x_mean
    y_centered = y_values - y_mean
    x_var = np.average(x_centered**2, weights=w_values)
    if pd.isna(x_var) or x_var == 0:
        return

    slope = np.average(x_centered * y_centered, weights=w_values) / x_var
    intercept = y_mean - slope * x_mean
    x_min = x_values.min()
    x_max = x_values.max()
    ax.plot(
        [x_min, x_max],
        [intercept + slope * x_min, intercept + slope * x_max],
        color="black",
        linewidth=1.2,
        alpha=0.35,
    )


def employment_size_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """Return employment bounds used for point-size scaling."""
    employment = df[EMP_COL].dropna()
    lower = employment.quantile(POINT_SIZE_LOWER)
    upper = employment.quantile(POINT_SIZE_UPPER)
    if pd.isna(lower) or pd.isna(upper) or lower == upper:
        return employment.min(), employment.max()
    return lower, upper


def employment_marker_sizes(
    employment: pd.Series,
    bounds: tuple[float, float],
) -> pd.Series:
    """Map employment to marker area, clipping extremes for legibility."""
    lower, upper = bounds
    if pd.isna(lower) or pd.isna(upper) or lower == upper:
        return pd.Series(POINT_SIZE_MIN, index=employment.index)

    clipped = employment.clip(lower=lower, upper=upper)
    scaled = (clipped - lower) / (upper - lower)
    return POINT_SIZE_MIN + (POINT_SIZE_MAX - POINT_SIZE_MIN) * scaled


def format_employment(x: float) -> str:
    """Format employment values compactly."""
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    return f"{x / 1_000:.0f}k"


def add_employment_legend(
    ax: plt.Axes,
    df: pd.DataFrame,
    bounds: tuple[float, float],
) -> None:
    """Add a point-size legend for imputed employment."""
    values = (
        df[EMP_COL].dropna().quantile([0.25, 0.50, 0.75]).round(-3).drop_duplicates()
    )
    handles = [
        ax.scatter(
            [],
            [],
            s=employment_marker_sizes(pd.Series([value]), bounds).iloc[0],
            color=POINT_COLOR,
            alpha=0.55,
            linewidths=0,
        )
        for value in values
    ]
    labels = [format_employment(value) for value in values]
    ax.legend(
        handles,
        labels,
        title="Employment",
        loc="lower right",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )


def add_panel_heading(
    ax: plt.Axes,
    platform_label: str,
    pearson: float,
    spearman: float,
    spearman_unweighted: float,
    n_obs: int,
) -> None:
    """Add a compact panel title with stats in parentheses."""
    ax.set_title(
        (
            f"$\\bf{{{platform_label}}}$\n"
            f"($r_w$={pearson:.2f}, $\\rho_w$={spearman:.2f}, "
            f"$\\rho$={spearman_unweighted:.2f}, n={n_obs:,})"
        ),
        fontsize=11,
        pad=7,
    )


def footer_text(width: int) -> str:
    """Return wrapped source and figure notes."""
    source = textwrap.fill(SOURCES, width=width, break_on_hyphens=False)
    note = textwrap.fill(FIGURE_NOTE, width=width, break_on_hyphens=False)
    return f"{source}\nNotes: {note}"


def apply_layout(fig: plt.Figure, layout: str) -> None:
    """Apply explicit spacing so titles, panels, and footer do not fight."""
    if layout == "wide":
        fig.subplots_adjust(left=0.055, right=0.985, top=0.78, bottom=0.34, wspace=0.20)
        fig.text(0.015, 0.045, footer_text(185), ha="left", va="bottom", fontsize=8)
    else:
        fig.subplots_adjust(left=0.16, right=0.97, top=0.89, bottom=0.16, hspace=0.50)
        fig.text(0.03, 0.025, footer_text(110), ha="left", va="bottom", fontsize=8)


def save_scatter_grid(
    df: pd.DataFrame,
    out_dir: Path,
    nonzero_only: bool = False,
    layout: str = "wide",
) -> Path:
    """Save one 3-panel wage scatter grid."""
    if layout == "wide":
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True)
    elif layout == "tall":
        fig, axes = plt.subplots(3, 1, figsize=(6.6, 11.5), sharex=True)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    axes = axes.ravel()
    size_bounds = employment_size_bounds(df)

    for ax, (platform, platform_label) in zip(axes, PLATFORMS.items(), strict=True):
        raw_col = net_col(platform)
        col = winsor_col(platform)
        plot_df = df[[WAGE_COL, EMP_COL, raw_col, col]].dropna()
        if nonzero_only:
            plot_df = plot_df[plot_df[raw_col] != 0]
        plot_df = plot_df[plot_df[EMP_COL] > 0]
        sizes = employment_marker_sizes(plot_df[EMP_COL], size_bounds)

        ax.scatter(
            plot_df[col],
            plot_df[WAGE_COL],
            s=sizes,
            color=POINT_COLOR,
            alpha=0.55,
            linewidths=0,
        )
        add_best_fit_line(ax, plot_df[col], plot_df[WAGE_COL], plot_df[EMP_COL])
        ax.axvline(0, color="black", linewidth=1, alpha=0.55)

        pearson = weighted_corr(plot_df[WAGE_COL], plot_df[col], plot_df[EMP_COL])
        spearman = weighted_rank_corr(
            plot_df[WAGE_COL],
            plot_df[col],
            plot_df[EMP_COL],
        )
        spearman_unweighted = plot_df[WAGE_COL].corr(plot_df[col], method="spearman")
        add_panel_heading(
            ax,
            platform_label,
            pearson,
            spearman,
            spearman_unweighted,
            len(plot_df),
        )
        ax.set_xlabel("Net Automation Usage")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_per_capita))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_wage))

    axes[0].set_ylabel("2024 OEWS annual mean wage")
    if layout == "tall":
        for ax in axes[1:]:
            ax.set_ylabel("2024 OEWS annual mean wage")
    add_employment_legend(axes[-1], df, size_bounds)

    title = "Net automation usage and wages by platform"
    if nonzero_only:
        title += " (nonzero usage)"
    fig.suptitle(title, fontsize=16)
    apply_layout(fig, layout)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_nonzero" if nonzero_only else ""
    layout_suffix = "_tall" if layout == "tall" else ""
    path = out_dir / f"net_automation_vs_wages{suffix}{layout_suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_notes(out_dir: Path) -> Path:
    """Save figure notes as a sidecar markdown file."""
    path = out_dir / "figure_notes.md"
    path.write_text(f"{SOURCE_DETAILS}\n{NOTE_TEXT}\n")
    return path


def write_output_codebook(out_dir: Path, winsor_lower: float, winsor_upper: float):
    """Write the codebook for the outputs in this directory."""
    net_cols = []
    for platform in PLATFORMS:
        net_cols.append(
            (
                net_col(platform),
                f"emp_{platform}_automation_count_pc minus "
                f"emp_{platform}_augmentation_count_pc from the 2024-OEWS AEI "
                "panel: net automation usage per worker, employment-"
                "apportioned.",
            )
        )
        net_cols.append(
            (
                winsor_col(platform),
                f"{net_col(platform)} winsorized at the "
                f"{winsor_lower:g}/{winsor_upper:g} quantiles.",
            )
        )
    update_codebook(
        out_dir / "codebook.md",
        section="net_automation_wages",
        title="Net automation wage analysis",
        source="analysis/net_automation_wages.py",
        intro=(
            "Built from occupations_aei_oews_2024.csv (see output/codebook.md "
            "for the input panel's variables)."
        ),
        files=[
            {
                "name": "occupation_net_automation_usage.csv",
                "description": (
                    "SOC 2010 panel of net automation usage (automation minus "
                    "augmentation, per worker)."
                ),
                "columns": [
                    ("soc_2010", "Six-digit SOC 2010 occupation code."),
                    ("title_2010", "SOC 2010 occupation title."),
                    (
                        "group_id",
                        "Connected-component ID of the SOC 2010<->2018 "
                        "crosswalk graph.",
                    ),
                    (
                        EMP_COL,
                        "OEWS employment allocated to SOC 2010, with missing "
                        "values filled with the median (from the input panel).",
                    ),
                    (WAGE_COL, "OEWS annual mean wage (May 2024)."),
                ]
                + net_cols,
            },
            {
                "name": "net_automation_wage_correlations.csv",
                "description": (
                    "Correlations between winsorized net automation usage and "
                    "the annual mean wage, one row per platform and sample."
                ),
                "columns": [
                    ("platform", "AEI platform: 1p_api, claude_ai, or total."),
                    ("platform_label", "Display label for the platform."),
                    ("net_column", "Winsorized net usage column analyzed."),
                    (
                        "nonzero_only",
                        "True when the sample is restricted to occupations "
                        "with nonzero net usage.",
                    ),
                    (
                        "weight_column",
                        f"Employment weight column ({EMP_COL}).",
                    ),
                    (
                        "winsor_lower_quantile / winsor_upper_quantile",
                        "Quantile levels used for winsorization.",
                    ),
                    (
                        "winsor_lower_value / winsor_upper_value",
                        "Data values at the winsorization quantiles.",
                    ),
                    ("n", "Occupations in the sample."),
                    (
                        "pearson / spearman",
                        "Employment-weighted correlations between the net "
                        f"column and {WAGE_COL} (Spearman is weighted Pearson "
                        "on ranks).",
                    ),
                    (
                        "pearson_unweighted / spearman_unweighted",
                        "Unweighted counterparts.",
                    ),
                    (
                        "mean_net / median_net",
                        "Unweighted mean and median of the winsorized net column.",
                    ),
                    (
                        "n_positive / n_negative / n_zero",
                        "Occupation counts by sign of the winsorized net column.",
                    ),
                ],
            },
        ],
    )
    log.info("Updated codebook: %s", out_dir / "codebook.md")


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

    write_output_codebook(args.out, args.winsor_lower, args.winsor_upper)

    notes_path = save_notes(args.out / "figures")
    log.info("Saved figure notes: %s", notes_path)

    path = save_scatter_grid(df, args.out / "figures")
    log.info("Saved figure: %s", path)
    path = save_scatter_grid(df, args.out / "figures", nonzero_only=True)
    log.info("Saved figure: %s", path)
    path = save_scatter_grid(df, args.out / "figures", layout="tall")
    log.info("Saved figure: %s", path)
    path = save_scatter_grid(
        df,
        args.out / "figures",
        nonzero_only=True,
        layout="tall",
    )
    log.info("Saved figure: %s", path)


if __name__ == "__main__":
    main()
