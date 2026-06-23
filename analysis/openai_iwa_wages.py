"""Analyze OpenAI IWA occupation usage against OEWS wages."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from eai.plot import apply_theme
from eai.utils import get_logger

log = get_logger(__name__)
apply_theme()

DEFAULT_PANEL = Path("output") / "openai_iwa_oews" / "openai_soc2018_mean_summary.csv"
DEFAULT_OUT = Path("output") / "openai_iwa_wages"
USAGE_COL = "mean_soc_2018_apportioned_share_of_messages"
WAGE_COL = "oews_a_mean"
EMP_COL = "oews_tot_emp_imputed"
PER_MILLION_WORKERS_COL = "mean_share_per_million_workers"
DEFAULT_WINSOR_LOWER = 0.01
DEFAULT_WINSOR_UPPER = 0.99
POINT_COLOR = "#4C72B0"
POINT_SIZE_MIN = 14
POINT_SIZE_MAX = 80
POINT_SIZE_LOWER = 0.05
POINT_SIZE_UPPER = 0.95

MEASURE_LABELS = {
    "us_all_messages_iwa_share": "All U.S. messages",
    "us_work_related_messages_iwa_share": "Work-related U.S. messages",
}

USAGE_VARIANTS = {
    USAGE_COL: {
        "label": "Mean apportioned share",
        "axis_label": "Mean apportioned share of messages",
        "formatter": "percent",
    },
    PER_MILLION_WORKERS_COL: {
        "label": "Mean apportioned share per million workers",
        "axis_label": "Mean apportioned share per million workers",
        "formatter": "per_million",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Analyze OpenAI IWA occupation usage against OEWS wages."
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_PANEL,
        help=f"Mean OpenAI IWA/OEWS occupation panel (default: {DEFAULT_PANEL}).",
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


def validate_winsor_bounds(lower: float, upper: float) -> None:
    """Validate winsorization quantile bounds."""
    if not 0 <= lower < upper <= 1:
        raise ValueError(
            "Expected winsorization bounds to satisfy "
            f"0 <= lower < upper <= 1, got lower={lower}, upper={upper}"
        )


def winsor_col(column: str) -> str:
    """Return the winsorized column name."""
    return f"{column}_winsorized"


def add_usage_columns(
    df: pd.DataFrame,
    winsor_lower: float,
    winsor_upper: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add employment-normalized usage and winsorized analysis columns."""
    df = df.copy()
    df[PER_MILLION_WORKERS_COL] = df[USAGE_COL] / df[EMP_COL] * 1_000_000

    rows = []
    for measure, sub in df.groupby("openai_measure"):
        for column in USAGE_VARIANTS:
            lower = sub[column].quantile(winsor_lower)
            upper = sub[column].quantile(winsor_upper)
            mask = df["openai_measure"] == measure
            df.loc[mask, winsor_col(column)] = df.loc[mask, column].clip(
                lower=lower,
                upper=upper,
            )
            rows.append(
                {
                    "openai_measure": measure,
                    "usage_column": column,
                    "winsorized_column": winsor_col(column),
                    "winsor_lower_quantile": winsor_lower,
                    "winsor_upper_quantile": winsor_upper,
                    "winsor_lower_value": lower,
                    "winsor_upper_value": upper,
                }
            )
    return df, pd.DataFrame(rows)


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


def build_correlations(
    df: pd.DataFrame,
    winsor_bounds: pd.DataFrame,
) -> pd.DataFrame:
    """Compute wage/usage correlations by OpenAI measure and usage variant."""
    rows = []
    for measure, measure_df in df.groupby("openai_measure"):
        for usage_col, meta in USAGE_VARIANTS.items():
            col = winsor_col(usage_col)
            data = measure_df[[WAGE_COL, EMP_COL, usage_col, col]].dropna()
            data = data[data[EMP_COL] > 0]
            bounds = winsor_bounds[
                (winsor_bounds["openai_measure"] == measure)
                & (winsor_bounds["usage_column"] == usage_col)
            ].iloc[0]
            rows.append(
                {
                    "openai_measure": measure,
                    "measure_label": MEASURE_LABELS.get(measure, measure),
                    "usage_variant": usage_col,
                    "usage_label": meta["label"],
                    "usage_column": col,
                    "weight_column": EMP_COL,
                    "wage_column": WAGE_COL,
                    "n": len(data),
                    "weighted_pearson": weighted_corr(
                        data[WAGE_COL], data[col], data[EMP_COL]
                    ),
                    "weighted_spearman": weighted_rank_corr(
                        data[WAGE_COL], data[col], data[EMP_COL]
                    ),
                    "pearson": data[WAGE_COL].corr(data[col], method="pearson"),
                    "spearman": data[WAGE_COL].corr(data[col], method="spearman"),
                    "mean_usage": data[usage_col].mean(),
                    "median_usage": data[usage_col].median(),
                    "winsor_lower_quantile": bounds["winsor_lower_quantile"],
                    "winsor_upper_quantile": bounds["winsor_upper_quantile"],
                    "winsor_lower_value": bounds["winsor_lower_value"],
                    "winsor_upper_value": bounds["winsor_upper_value"],
                }
            )
    return pd.DataFrame(rows)


def format_wage(x: float, _pos: int) -> str:
    """Format wage ticks as thousands of dollars."""
    return f"${x / 1000:.0f}k"


def format_percent(x: float, _pos: int) -> str:
    """Format share ticks as percentages."""
    return f"{x * 100:.1f}%"


def format_per_million(x: float, _pos: int) -> str:
    """Format per-worker usage ticks compactly."""
    return f"{x:.3g}"


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


def add_best_fit_line(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    weights: pd.Series,
) -> None:
    """Draw a light employment-weighted OLS line for y on x."""
    data = pd.concat([x, y, weights], axis=1).dropna()
    data = data[data.iloc[:, 2] > 0]
    if len(data) < 2:
        return

    x_values = data.iloc[:, 0].astype(float)
    y_values = data.iloc[:, 1].astype(float)
    w_values = data.iloc[:, 2].astype(float)
    if x_values.nunique() < 2:
        return

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


def add_panel_heading(
    ax: plt.Axes,
    label: str,
    weighted_pearson: float,
    weighted_spearman: float,
    spearman: float,
    n_obs: int,
) -> None:
    """Add a compact panel title with stats in parentheses."""
    stats = (
        f"(weighted r={weighted_pearson:.2f}, "
        f"weighted rho={weighted_spearman:.2f}, "
        f"rho={spearman:.2f}, n={n_obs:,})"
    )
    ax.set_title(
        f"{label}\n{stats}",
        fontsize=11,
        pad=7,
        fontweight="bold",
    )


def figure_note(width: int) -> str:
    """Return wrapped figure notes."""
    text = (
        "Sources: OpenAI Signals IWA monthly files, O*NET 30.2, and BLS OEWS "
        "May 2024 national estimates. Usage is employment-apportioned from IWA "
        "to SOC 2018, averaged across months, and winsorized p1/p99 within each "
        "OpenAI measure for plotting and correlations. Marker size is imputed "
        "employment clipped p5/p95. Gray line is employment-weighted OLS."
    )
    return textwrap.fill(text, width=width, break_on_hyphens=False)


def save_scatter(
    df: pd.DataFrame,
    correlations: pd.DataFrame,
    usage_col: str,
    out_dir: Path,
) -> Path:
    """Save a two-panel scatter of usage against wages."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    size_bounds = employment_size_bounds(df)
    plot_col = winsor_col(usage_col)
    meta = USAGE_VARIANTS[usage_col]

    for ax, (measure, sub) in zip(
        axes,
        df.groupby("openai_measure"),
        strict=True,
    ):
        plot_df = sub[[plot_col, WAGE_COL, EMP_COL]].dropna()
        plot_df = plot_df[plot_df[EMP_COL] > 0]
        sizes = employment_marker_sizes(plot_df[EMP_COL], size_bounds)
        ax.scatter(
            plot_df[plot_col],
            plot_df[WAGE_COL],
            s=sizes,
            color=POINT_COLOR,
            alpha=0.55,
            linewidths=0,
        )
        add_best_fit_line(
            ax,
            plot_df[plot_col],
            plot_df[WAGE_COL],
            plot_df[EMP_COL],
        )
        corr = correlations[
            (correlations["openai_measure"] == measure)
            & (correlations["usage_variant"] == usage_col)
        ].iloc[0]
        add_panel_heading(
            ax,
            MEASURE_LABELS.get(measure, measure),
            corr["weighted_pearson"],
            corr["weighted_spearman"],
            corr["spearman"],
            int(corr["n"]),
        )
        ax.set_xlabel(meta["axis_label"])
        if meta["formatter"] == "percent":
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_percent))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_per_million))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_wage))

    axes[0].set_ylabel("2024 OEWS annual mean wage")
    fig.suptitle(f"{meta['label']} and wages", fontsize=15)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.78, bottom=0.28, wspace=0.15)
    fig.text(0.02, 0.035, figure_note(150), ha="left", va="bottom", fontsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "mean_share" if usage_col == USAGE_COL else "mean_share_per_million_workers"
    path = out_dir / f"{stem}_vs_wages.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    panel_path: Path,
    panel: pd.DataFrame,
    correlations: pd.DataFrame,
    figure_paths: list[Path],
) -> Path:
    """Write a compact Markdown report for the analysis outputs."""
    corr_view = correlations[
        [
            "measure_label",
            "usage_label",
            "n",
            "weighted_pearson",
            "weighted_spearman",
            "pearson",
            "spearman",
        ]
    ].copy()
    corr_view = corr_view.rename(
        columns={
            "measure_label": "measure",
            "usage_label": "usage",
        }
    )
    summary = (
        panel.groupby("openai_measure", as_index=False)
        .agg(
            occupations=("soc_2018", "nunique"),
            min_months=("n_months", "min"),
            max_months=("n_months", "max"),
            first_month=("first_month", "min"),
            last_month=("last_month", "max"),
        )
        .replace({"openai_measure": MEASURE_LABELS})
    )

    def markdown_table(df: pd.DataFrame) -> str:
        rows = [
            "| " + " | ".join(df.columns) + " |",
            "| " + " | ".join(["---"] * len(df.columns)) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    figures = "\n".join(f"- `{path}`" for path in figure_paths)
    report = f"""# OpenAI IWA Usage and OEWS Wages

Input panel: `{panel_path}`

This analysis uses the across-month mean OpenAI IWA occupation usage file and compares
usage with 2024 OEWS annual mean wages. It reports both employment-weighted and
unweighted Pearson/Spearman correlations.

## Coverage

{markdown_table(summary)}

## Correlations

{markdown_table(corr_view)}

## Outputs

- `openai_usage_wage_analysis_panel.csv`
- `openai_usage_wage_correlations.csv`
- `winsor_bounds.csv`
{figures}

## Notes

- `Mean apportioned share` is the monthly IWA-to-SOC 2018 usage share averaged
  across all available months.
- `Mean apportioned share per million workers` divides that share by imputed
  OEWS employment and multiplies by one million.
- Correlations use p1/p99 winsorized usage values within each OpenAI measure
  and usage variant.
- Weighted correlations use `oews_tot_emp_imputed` as weights.
"""
    path = out_dir / "openai_usage_wage_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    """Run the analysis."""
    args = parse_args()
    validate_winsor_bounds(args.winsor_lower, args.winsor_upper)
    if not args.panel.exists():
        raise FileNotFoundError(f"Missing input panel: {args.panel}")

    panel = pd.read_csv(args.panel)
    log.info("Loaded %s: %d rows", args.panel, len(panel))

    panel, winsor_bounds = add_usage_columns(
        panel,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
    )
    correlations = build_correlations(panel, winsor_bounds)

    args.out.mkdir(parents=True, exist_ok=True)
    panel_path = args.out / "openai_usage_wage_analysis_panel.csv"
    corr_path = args.out / "openai_usage_wage_correlations.csv"
    bounds_path = args.out / "winsor_bounds.csv"
    panel.to_csv(panel_path, index=False)
    correlations.to_csv(corr_path, index=False)
    winsor_bounds.to_csv(bounds_path, index=False)
    log.info("Saved analysis panel: %s", panel_path)
    log.info("Saved correlations: %s", corr_path)
    log.info("Saved winsor bounds: %s", bounds_path)

    figure_dir = args.out / "figures"
    figure_paths = [
        save_scatter(panel, correlations, USAGE_COL, figure_dir),
        save_scatter(panel, correlations, PER_MILLION_WORKERS_COL, figure_dir),
    ]
    for path in figure_paths:
        log.info("Saved figure: %s", path)

    report_path = write_report(
        args.out,
        args.panel,
        panel,
        correlations,
        figure_paths,
    )
    log.info("Saved report: %s", report_path)


if __name__ == "__main__":
    main()
