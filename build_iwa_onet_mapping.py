# ruff: noqa: E501
"""Build IWA to O*NET-SOC mapping artifacts from O*NET 30.2 Excel files."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INPUT_DIR = Path("input") / "db_30_2_excel"
OUTPUT_DIR = Path("output") / "onet" / "iwa_mapping"
DOC_PDF = INPUT_DIR / "db_30_2_dictionary.pdf"

WEIGHT_COLUMNS = [
    "occupation_weight_within_iwa_task_count",
    "occupation_weight_within_iwa_task_dwa_links",
    "occupation_weight_within_iwa_dwa_count",
    "occupation_weight_within_iwa_core_task_count",
    "occupation_weight_within_iwa_classified_task_count",
]

WEIGHT_LABELS = {
    "occupation_weight_within_iwa_task_count": "Distinct task count",
    "occupation_weight_within_iwa_task_dwa_links": "Task-DWA edge count",
    "occupation_weight_within_iwa_dwa_count": "Distinct DWA count",
    "occupation_weight_within_iwa_core_task_count": "Core task count",
    "occupation_weight_within_iwa_classified_task_count": "Classified task count",
}

STRING_COLUMNS = {
    "O*NET-SOC Code",
    "Title",
    "IWA ID",
    "IWA Title",
    "DWA ID",
    "DWA Title",
    "Task",
    "Task Type",
    "Date",
    "Domain Source",
}


def read_excel(name: str) -> pd.DataFrame:
    """Read one workbook and preserve identifier columns as trimmed strings."""
    df = pd.read_excel(INPUT_DIR / name)
    for column in STRING_COLUMNS.intersection(df.columns):
        df[column] = df[column].astype("string").str.strip()
    return df


def clean_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert O*NET source headers to stable snake_case output columns."""
    return df.rename(
        columns={
            "O*NET-SOC Code": "onet_soc_code",
            "Title": "occupation_title",
            "IWA ID": "iwa_id",
            "IWA Title": "iwa_title",
            "DWA ID": "dwa_id",
            "DWA Title": "dwa_title",
            "Task ID": "task_id",
            "Task": "task",
            "Task Type": "task_type",
            "Date": "date",
            "Domain Source": "domain_source",
        }
    )


def join_unique(values: pd.Series) -> str:
    items = sorted({str(value) for value in values.dropna() if str(value)})
    return "; ".join(items)


def parse_periods(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.dropna(), format="%m/%Y", errors="coerce").dropna()


def min_period(values: pd.Series) -> str:
    parsed = parse_periods(values)
    if parsed.empty:
        return ""
    return parsed.min().strftime("%Y-%m")


def max_period(values: pd.Series) -> str:
    parsed = parse_periods(values)
    if parsed.empty:
        return ""
    return parsed.max().strftime("%Y-%m")


def safe_within_iwa_weight(df: pd.DataFrame, column: str) -> pd.Series:
    denominator = df.groupby("iwa_id")[column].transform("sum")
    return df[column].div(denominator).where(denominator != 0)


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    return path


def validate_inputs(
    iwa: pd.DataFrame,
    dwa: pd.DataFrame,
    task_statements: pd.DataFrame,
    tasks_to_dwas: pd.DataFrame,
) -> dict[str, object]:
    checks: dict[str, object] = {
        "iwa_ids_unique": bool(iwa["IWA ID"].is_unique),
        "dwa_ids_unique": bool(dwa["DWA ID"].is_unique),
        "task_ids_unique": bool(task_statements["Task ID"].is_unique),
        "dwa_iwa_missing_from_iwa_reference": len(
            set(dwa["IWA ID"]) - set(iwa["IWA ID"])
        ),
        "iwa_missing_from_dwa_reference": len(set(iwa["IWA ID"]) - set(dwa["IWA ID"])),
        "tasks_to_dwas_dwa_missing_from_dwa_reference": len(
            set(tasks_to_dwas["DWA ID"]) - set(dwa["DWA ID"])
        ),
        "tasks_to_dwas_task_missing_from_task_statements": len(
            set(tasks_to_dwas["Task ID"]) - set(task_statements["Task ID"])
        ),
    }
    failures = {
        key: value
        for key, value in checks.items()
        if key.endswith("_missing_from_iwa_reference")
        or key.endswith("_missing_from_dwa_reference")
        or key.endswith("_missing_from_task_statements")
    }
    failures = {key: value for key, value in failures.items() if value}
    if failures:
        raise ValueError(f"Referential integrity check failed: {failures}")
    if not checks["iwa_ids_unique"] or not checks["dwa_ids_unique"]:
        raise ValueError("Expected unique IWA and DWA identifiers.")
    return checks


def build_task_mapping(
    iwa: pd.DataFrame,
    dwa: pd.DataFrame,
    task_statements: pd.DataFrame,
    tasks_to_dwas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dwa_bridge = dwa[["IWA ID", "IWA Title", "DWA ID", "DWA Title"]].drop_duplicates()
    task_meta = task_statements[
        [
            "O*NET-SOC Code",
            "Task ID",
            "Task Type",
            "Incumbents Responding",
            "Date",
            "Domain Source",
        ]
    ].rename(
        columns={
            "Date": "Task Statement Date",
            "Domain Source": "Task Statement Domain Source",
        }
    )
    detail = tasks_to_dwas[
        [
            "O*NET-SOC Code",
            "Title",
            "Task ID",
            "Task",
            "DWA ID",
            "Date",
            "Domain Source",
        ]
    ].merge(dwa_bridge, on="DWA ID", how="left", validate="m:1")
    detail = detail.merge(
        task_meta,
        on=["O*NET-SOC Code", "Task ID"],
        how="left",
        validate="m:1",
        indicator="_task_statement_merge",
    )
    if detail["IWA ID"].isna().any():
        raise ValueError("Tasks to DWAs contains DWA IDs not found in DWA Reference.")
    if (detail["_task_statement_merge"] != "both").any():
        raise ValueError(
            "Tasks to DWAs contains task IDs not found in Task Statements."
        )

    detail = clean_output_columns(detail.drop(columns="_task_statement_merge"))
    detail = detail[
        [
            "iwa_id",
            "iwa_title",
            "onet_soc_code",
            "occupation_title",
            "task_id",
            "task",
            "task_type",
            "dwa_id",
            "dwa_title",
            "Incumbents Responding",
            "date",
            "domain_source",
            "Task Statement Date",
            "Task Statement Domain Source",
        ]
    ].rename(
        columns={
            "Incumbents Responding": "incumbents_responding",
            "Task Statement Date": "task_statement_date",
            "Task Statement Domain Source": "task_statement_domain_source",
        }
    )
    detail["task_type_clean"] = detail["task_type"].fillna("Unclassified")

    group_keys = ["iwa_id", "onet_soc_code"]
    task_counts = (
        detail.drop_duplicates(group_keys + ["task_id"])
        .groupby(group_keys + ["task_type_clean"], dropna=False)["task_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["Core", "Supplemental", "Unclassified"]:
        if column not in task_counts.columns:
            task_counts[column] = 0
    task_counts = task_counts.rename(
        columns={
            "Core": "core_task_count",
            "Supplemental": "supplemental_task_count",
            "Unclassified": "unclassified_task_count",
        }
    )

    summary = (
        detail.groupby(group_keys, dropna=False)
        .agg(
            iwa_title=("iwa_title", "first"),
            occupation_title=("occupation_title", "first"),
            task_count=("task_id", "nunique"),
            task_dwa_link_count=("task_id", "size"),
            dwa_count=("dwa_id", "nunique"),
            first_mapping_update=("date", min_period),
            latest_mapping_update=("date", max_period),
            domain_sources=("domain_source", join_unique),
        )
        .reset_index()
        .merge(task_counts, on=group_keys, how="left", validate="1:1")
    )
    summary["classified_task_count"] = (
        summary["core_task_count"] + summary["supplemental_task_count"]
    )
    summary["occupation_weight_within_iwa_task_count"] = safe_within_iwa_weight(
        summary, "task_count"
    )
    summary["occupation_weight_within_iwa_task_dwa_links"] = safe_within_iwa_weight(
        summary, "task_dwa_link_count"
    )
    summary["occupation_weight_within_iwa_dwa_count"] = safe_within_iwa_weight(
        summary, "dwa_count"
    )
    summary["occupation_weight_within_iwa_core_task_count"] = safe_within_iwa_weight(
        summary, "core_task_count"
    )
    summary["occupation_weight_within_iwa_classified_task_count"] = (
        safe_within_iwa_weight(summary, "classified_task_count")
    )
    summary["mapping_method"] = "task_dwa_reference"
    summary["notes"] = (
        "O*NET 30.2 Tasks to DWAs links task statements to DWAs and consequently "
        "to O*NET-SOC occupations; DWA Reference links each DWA to exactly one IWA."
    )

    missing_iwas = set(iwa["IWA ID"]) - set(summary["iwa_id"])
    if missing_iwas:
        raise ValueError(f"Task-derived mapping omitted IWAs: {sorted(missing_iwas)}")

    summary = summary[
        [
            "iwa_id",
            "iwa_title",
            "onet_soc_code",
            "occupation_title",
            "task_count",
            "task_dwa_link_count",
            "dwa_count",
            "core_task_count",
            "supplemental_task_count",
            "unclassified_task_count",
            "classified_task_count",
            "first_mapping_update",
            "latest_mapping_update",
            "domain_sources",
            *WEIGHT_COLUMNS,
            "mapping_method",
            "notes",
        ]
    ]

    return detail.sort_values(group_keys + ["dwa_id", "task_id"]), summary.sort_values(
        group_keys
    )


def build_iwa_occupation_counts(task_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        task_summary.groupby(["iwa_id", "iwa_title"], as_index=False)
        .agg(
            occupation_count=("onet_soc_code", "nunique"),
            task_count=("task_count", "sum"),
            task_dwa_link_count=("task_dwa_link_count", "sum"),
            dwa_count=("dwa_count", "sum"),
            core_task_count=("core_task_count", "sum"),
            supplemental_task_count=("supplemental_task_count", "sum"),
            unclassified_task_count=("unclassified_task_count", "sum"),
            classified_task_count=("classified_task_count", "sum"),
        )
        .sort_values(["occupation_count", "iwa_id"], ascending=[False, True])
    )


def build_occupation_iwa_counts(task_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        task_summary.groupby(["onet_soc_code", "occupation_title"], as_index=False)
        .agg(
            iwa_count=("iwa_id", "nunique"),
            task_count=("task_count", "sum"),
            task_dwa_link_count=("task_dwa_link_count", "sum"),
            dwa_count=("dwa_count", "sum"),
            core_task_count=("core_task_count", "sum"),
            supplemental_task_count=("supplemental_task_count", "sum"),
            unclassified_task_count=("unclassified_task_count", "sum"),
            classified_task_count=("classified_task_count", "sum"),
        )
        .sort_values(["iwa_count", "onet_soc_code"], ascending=[False, True])
    )


def build_iwa_occupation_links(
    task_summary: pd.DataFrame,
    iwa_counts: pd.DataFrame,
    occupation_counts: pd.DataFrame,
) -> pd.DataFrame:
    links = task_summary[
        [
            "iwa_id",
            "iwa_title",
            "onet_soc_code",
            "occupation_title",
            "task_count",
            "task_dwa_link_count",
            "dwa_count",
            "core_task_count",
            "supplemental_task_count",
            "unclassified_task_count",
            "classified_task_count",
            *WEIGHT_COLUMNS,
        ]
    ].rename(
        columns={
            "task_count": "link_task_count",
            "task_dwa_link_count": "link_task_dwa_link_count",
            "dwa_count": "link_dwa_count",
            "core_task_count": "link_core_task_count",
            "supplemental_task_count": "link_supplemental_task_count",
            "unclassified_task_count": "link_unclassified_task_count",
            "classified_task_count": "link_classified_task_count",
        }
    )
    links = links.merge(
        iwa_counts[["iwa_id", "occupation_count"]].rename(
            columns={"occupation_count": "occupation_count_for_iwa"}
        ),
        on="iwa_id",
        how="left",
        validate="m:1",
    )
    links = links.merge(
        occupation_counts[["onet_soc_code", "iwa_count"]].rename(
            columns={"iwa_count": "iwa_count_for_occupation"}
        ),
        on="onet_soc_code",
        how="left",
        validate="m:1",
    )
    return links[
        [
            "iwa_id",
            "iwa_title",
            "onet_soc_code",
            "occupation_title",
            "link_dwa_count",
            "iwa_count_for_occupation",
            "occupation_count_for_iwa",
            "link_task_count",
            "link_task_dwa_link_count",
            "link_core_task_count",
            "link_supplemental_task_count",
            "link_unclassified_task_count",
            "link_classified_task_count",
            *WEIGHT_COLUMNS,
        ]
    ].sort_values(["iwa_id", "onet_soc_code"])


def build_weight_summary(task_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in WEIGHT_COLUMNS:
        values = pd.to_numeric(task_summary[column], errors="coerce")
        rows.append(
            {
                "weight_column": column,
                "basis": WEIGHT_LABELS[column],
                "n": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "zero_count": int((values == 0).sum()),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "p25": values.quantile(0.25),
                "median": values.quantile(0.50),
                "p75": values.quantile(0.75),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def build_weight_correlations(
    task_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = task_summary[WEIGHT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    rows = []
    for method, matrix in [("pearson", pearson), ("spearman", spearman)]:
        for idx, left in enumerate(WEIGHT_COLUMNS):
            for right in WEIGHT_COLUMNS[idx + 1 :]:
                rows.append(
                    {
                        "method": method,
                        "weight_1": left,
                        "weight_2": right,
                        "correlation": matrix.loc[left, right],
                    }
                )
    return pd.DataFrame(rows), pearson, spearman


def save_weight_scatter(task_summary: pd.DataFrame, output_path: Path) -> Path:
    x_col = "occupation_weight_within_iwa_task_count"
    y_cols = [column for column in WEIGHT_COLUMNS if column != x_col]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, y_col in zip(axes.flatten(), y_cols, strict=True):
        plot_df = (
            task_summary[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        )
        ax.scatter(plot_df[x_col], plot_df[y_col], s=9, alpha=0.25, linewidths=0)
        corr = plot_df[x_col].corr(plot_df[y_col])
        min_value = float(min(plot_df[x_col].min(), plot_df[y_col].min()))
        max_value = float(max(plot_df[x_col].max(), plot_df[y_col].max()))
        ax.plot([min_value, max_value], [min_value, max_value], color="#444", lw=1)
        ax.set_xlabel(WEIGHT_LABELS[x_col])
        ax.set_ylabel(WEIGHT_LABELS[y_col])
        ax.set_title(f"Pearson r = {corr:.3f}")
    fig.suptitle("Allocation weights compared with distinct-task weighting")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_weight_correlation_heatmap(
    correlations: pd.DataFrame, output_path: Path
) -> Path:
    labels = [WEIGHT_LABELS[column] for column in correlations.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(correlations.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pearson correlations across allocation weights")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(
    df: pd.DataFrame, columns: list[str], limit: int | None = None
) -> str:
    table = df[columns].copy()
    if limit is not None:
        table = table.head(limit)
    rows = []
    headers = [column.replace("_", " ") for column in columns]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def distribution_row(df: pd.DataFrame, column: str) -> list[object]:
    values = pd.to_numeric(df[column], errors="coerce")
    return [
        column,
        int(values.count()),
        values.mean(),
        values.min(),
        values.quantile(0.25),
        values.median(),
        values.quantile(0.75),
        values.max(),
    ]


def write_markdown_report(
    paths: dict[str, Path],
    counts: dict[str, object],
    checks: dict[str, object],
    task_summary: pd.DataFrame,
    iwa_occupation_links: pd.DataFrame,
    iwa_counts: pd.DataFrame,
    occupation_counts: pd.DataFrame,
    weight_summary: pd.DataFrame,
    pearson: pd.DataFrame,
) -> Path:
    coverage_distribution = pd.DataFrame(
        [
            distribution_row(iwa_counts, "occupation_count"),
            distribution_row(occupation_counts, "iwa_count"),
        ],
        columns=["metric", "n", "mean", "min", "p25", "median", "p75", "max"],
    )
    correlation_highlights = []
    default = "occupation_weight_within_iwa_task_count"
    for column in WEIGHT_COLUMNS:
        if column == default:
            continue
        correlation_highlights.append(
            {
                "comparison": f"{WEIGHT_LABELS[default]} vs {WEIGHT_LABELS[column]}",
                "pearson_r": pearson.loc[default, column],
            }
        )
    correlation_highlights = pd.DataFrame(correlation_highlights)

    report = f"""# IWA to O*NET-SOC Mapping - O*NET 30.2

Generated {date.today().isoformat()} from `{
        INPUT_DIR
    }`. This report is restricted to `O*NET-SOC Code` occupation identifiers and stays pinned to O*NET 30.2.

## Mapping Path

Use `{paths["task_summary"].name}` for IWA to O*NET-SOC analysis:

`IWA Reference` -> `DWA Reference` -> `Tasks to DWAs` -> `O*NET-SOC Code`

## Deliverables

{
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "artifact": paths["task_summary"].name,
                        "rows": len(task_summary),
                        "description": "IWA to O*NET-SOC mapping, one row per IWA/occupation pair.",
                    },
                    {
                        "artifact": paths["iwa_occupation_links"].name,
                        "rows": len(iwa_occupation_links),
                        "description": "Slim IWA-to-occupation edge table with DWA and IWA coverage counts.",
                    },
                    {
                        "artifact": paths["task_detail"].name,
                        "rows": counts["task_detail_rows"],
                        "description": "Audit table with one row per task-to-DWA-to-IWA link.",
                    },
                    {
                        "artifact": paths["iwa_counts"].name,
                        "rows": len(iwa_counts),
                        "description": "Number of occupations mapped to each IWA.",
                    },
                    {
                        "artifact": paths["occupation_counts"].name,
                        "rows": len(occupation_counts),
                        "description": "Number of IWAs mapped to each O*NET-SOC occupation.",
                    },
                    {
                        "artifact": paths["weight_summary"].name,
                        "rows": len(weight_summary),
                        "description": "Summary statistics for the allocation weight columns.",
                    },
                    {
                        "artifact": paths["weight_correlations"].name,
                        "rows": len(pd.read_csv(paths["weight_correlations"])),
                        "description": "Pearson and Spearman pairwise correlations among allocation weights.",
                    },
                    {
                        "artifact": paths["pearson_matrix"].name,
                        "rows": len(WEIGHT_COLUMNS),
                        "description": "Pearson correlation matrix for allocation weights.",
                    },
                    {
                        "artifact": paths["spearman_matrix"].name,
                        "rows": len(WEIGHT_COLUMNS),
                        "description": "Spearman correlation matrix for allocation weights.",
                    },
                    {
                        "artifact": paths["weight_scatter"].name,
                        "rows": "",
                        "description": "Scatter plots comparing allocation weights.",
                    },
                    {
                        "artifact": paths["weight_heatmap"].name,
                        "rows": "",
                        "description": "Pearson correlation heatmap for allocation weights.",
                    },
                ]
            ),
            ["artifact", "rows", "description"],
        )
    }

## Documentation Basis

Local source: `{
        DOC_PDF
    }`. The local PDF identifies itself as **Data Dictionary - O*NET 30.2 Database**, created February 11, 2026, with 110 pages.

- PDF page 31: `Task Statements` maps O*NET-SOC occupations to task statements and defines the `Task Type` field.
- PDF page 47: `IWA Reference` provides Intermediate Work Activity identifiers.
- PDF page 48: `DWA Reference` links each DWA to exactly one IWA.
- PDF page 49: `Tasks to DWAs` maps task statements to DWAs and consequently to O*NET-SOC occupations.

Internet check: the official O*NET 30.2 online data dictionary confirms the same relationship across [Task Statements](https://www.onetcenter.org/dictionary/30.2/excel/task_statements.html), [IWA Reference](https://www.onetcenter.org/dictionary/30.2/excel/iwa_reference.html), [DWA Reference](https://www.onetcenter.org/dictionary/30.2/excel/dwa_reference.html), and [Tasks to DWAs](https://www.onetcenter.org/dictionary/30.2/excel/tasks_to_dwas.html).

## Source Counts

{
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "source": "IWA Reference",
                        "rows": f"{counts['iwa_rows']:,}",
                        "coverage": "All represented in the output mapping",
                    },
                    {
                        "source": "DWA Reference",
                        "rows": f"{counts['dwa_rows']:,}",
                        "coverage": f"{counts['dwa_iwas']:,} IWAs",
                    },
                    {
                        "source": "Tasks to DWAs",
                        "rows": f"{counts['tasks_to_dwas_rows']:,}",
                        "coverage": f"{counts['tasks_to_dwas_occupations']:,} O*NET-SOC occupations",
                    },
                    {
                        "source": "Task Statements",
                        "rows": f"{counts['task_statement_rows']:,}",
                        "coverage": f"{counts['task_statement_occupations']:,} O*NET-SOC occupations",
                    },
                ]
            ),
            ["source", "rows", "coverage"],
        )
    }

## Validation Results

- Every IWA ID is unique: **{checks["iwa_ids_unique"]}**.
- Every DWA ID is unique: **{checks["dwa_ids_unique"]}**.
- Every Task ID is unique in Task Statements: **{checks["task_ids_unique"]}**.
- DWAs missing from IWA Reference: **{checks["dwa_iwa_missing_from_iwa_reference"]}**.
- IWAs missing from DWA Reference: **{checks["iwa_missing_from_dwa_reference"]}**.
- `Tasks to DWAs` DWA IDs missing from DWA Reference: **{
        checks["tasks_to_dwas_dwa_missing_from_dwa_reference"]
    }**.
- `Tasks to DWAs` task IDs missing from Task Statements: **{
        checks["tasks_to_dwas_task_missing_from_task_statements"]
    }**.
- Task-DWA detail rows with blank `Task Type`, retained as `Unclassified`: **{
        counts["unclassified_task_detail_rows"]:,}**.

## Mapping Results

- IWA/O*NET-SOC pairs: **{len(task_summary):,}**.
- Unique IWAs covered: **{task_summary["iwa_id"].nunique():,}**.
- Unique O*NET-SOC occupations covered: **{task_summary["onet_soc_code"].nunique():,}**.
- Task-to-DWA-to-IWA detail rows: **{counts["task_detail_rows"]:,}**.

## IWA-to-Occupation Link Table

`{
        paths["iwa_occupation_links"].name
    }` is the compact edge table for network or crosswalk use. Each row is one IWA/O*NET-SOC link.

- `link_dwa_count`: number of distinct DWAs connecting that IWA and occupation.
- `iwa_count_for_occupation`: number of distinct IWAs linked to that occupation.
- `occupation_count_for_iwa`: number of distinct occupations linked to that IWA.

{
        markdown_table(
            iwa_occupation_links,
            [
                "iwa_id",
                "iwa_title",
                "onet_soc_code",
                "occupation_title",
                "link_dwa_count",
                "iwa_count_for_occupation",
                "occupation_count_for_iwa",
            ],
            limit=10,
        )
    }

## Occupation And IWA Coverage

The IWA side asks how many occupations each IWA touches. The occupation side asks how many IWAs appear in each occupation.

{
        markdown_table(
            coverage_distribution,
            ["metric", "n", "mean", "min", "p25", "median", "p75", "max"],
        )
    }

### IWAs With The Most Occupations

{
        markdown_table(
            iwa_counts,
            [
                "iwa_id",
                "iwa_title",
                "occupation_count",
                "task_count",
                "task_dwa_link_count",
            ],
            limit=10,
        )
    }

### Occupations With The Most IWAs

{
        markdown_table(
            occupation_counts,
            [
                "onet_soc_code",
                "occupation_title",
                "iwa_count",
                "task_count",
                "task_dwa_link_count",
            ],
            limit=10,
        )
    }

## Core And Supplemental Tasks

O*NET's `Task Type` field distinguishes tasks that are central to an occupation from tasks that are still associated with the occupation but less central. A **Core** task is critical to the occupation: O*NET defines this as task relevance of at least 67% and mean importance of at least 3.0. A **Supplemental** task is less relevant and/or less important: either relevance is at least 67% but mean importance is below 3.0, or relevance is below 67% regardless of importance. Some rows in this release have a blank Task Type; these are retained as `Unclassified` so the mapping does not silently drop valid task-DWA links.

## Allocation Weight Columns

Allocation here means taking an IWA-level quantity, such as a share of messages coded to an IWA, and distributing it across O*NET-SOC occupations. Every `occupation_weight_within_iwa_...` column is normalized within each IWA, so weights sum to 1 across occupations for that IWA when the denominator is nonzero.

{
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "column": "occupation_weight_within_iwa_task_count",
                        "basis": "Distinct task count",
                        "when_to_use": "Best default when one task should count once even if it has multiple DWA links.",
                    },
                    {
                        "column": "occupation_weight_within_iwa_task_dwa_links",
                        "basis": "Task-DWA edge count",
                        "when_to_use": "Use when multiple DWA links on the same task should count as stronger evidence.",
                    },
                    {
                        "column": "occupation_weight_within_iwa_dwa_count",
                        "basis": "Distinct DWA breadth",
                        "when_to_use": "Use when breadth of detailed activities matters more than task count.",
                    },
                    {
                        "column": "occupation_weight_within_iwa_core_task_count",
                        "basis": "Core-task-only count",
                        "when_to_use": "Use for a conservative mapping focused on tasks O*NET marks critical.",
                    },
                    {
                        "column": "occupation_weight_within_iwa_classified_task_count",
                        "basis": "Core plus supplemental count",
                        "when_to_use": "Use if you want official task categories but want to exclude unclassified rows.",
                    },
                ]
            ),
            ["column", "basis", "when_to_use"],
        )
    }

## Weight Summary

{
        markdown_table(
            weight_summary,
            [
                "weight_column",
                "basis",
                "n",
                "missing",
                "zero_count",
                "mean",
                "median",
                "p75",
                "max",
            ],
        )
    }

## Weight Correlations

The table below compares each alternate weighting scheme with the distinct-task-count default. Full Pearson and Spearman results are in `{
        paths["weight_correlations"].name
    }`.

{markdown_table(correlation_highlights, ["comparison", "pearson_r"])}

![Allocation weight scatter plots]({paths["weight_scatter"].name})

![Allocation weight correlation heatmap]({paths["weight_heatmap"].name})

## Outstanding Questions

- Which allocation column should downstream analysis use as the default? I would start with `occupation_weight_within_iwa_task_count` unless we want task-DWA link multiplicity to carry extra signal.
- Should downstream results include all task evidence, or should they report a core-task-only sensitivity using `occupation_weight_within_iwa_core_task_count`?
- Should the OAI `Other IWA` bucket be excluded, manually reviewed, or apportioned across known IWAs before occupation-level aggregation?
"""
    report_path = OUTPUT_DIR / "iwa_onet_mapping_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    iwa = read_excel("IWA Reference.xlsx")
    dwa = read_excel("DWA Reference.xlsx")
    task_statements = read_excel("Task Statements.xlsx")
    tasks_to_dwas = read_excel("Tasks to DWAs.xlsx")

    checks = validate_inputs(iwa, dwa, task_statements, tasks_to_dwas)
    task_detail, task_summary = build_task_mapping(
        iwa, dwa, task_statements, tasks_to_dwas
    )
    iwa_counts = build_iwa_occupation_counts(task_summary)
    occupation_counts = build_occupation_iwa_counts(task_summary)
    iwa_occupation_links = build_iwa_occupation_links(
        task_summary, iwa_counts, occupation_counts
    )
    weight_summary = build_weight_summary(task_summary)
    weight_correlations, pearson, spearman = build_weight_correlations(task_summary)

    paths = {
        "task_detail": write_csv(task_detail, "iwa_to_onet_soc_via_tasks_detail.csv"),
        "task_summary": write_csv(task_summary, "iwa_to_onet_soc_via_tasks.csv"),
        "iwa_occupation_links": write_csv(
            iwa_occupation_links, "iwa_occupation_links.csv"
        ),
        "iwa_counts": write_csv(iwa_counts, "iwa_occupation_counts.csv"),
        "occupation_counts": write_csv(occupation_counts, "occupation_iwa_counts.csv"),
        "weight_summary": write_csv(weight_summary, "iwa_weight_summary.csv"),
        "weight_correlations": write_csv(
            weight_correlations, "iwa_weight_correlations.csv"
        ),
        "pearson_matrix": write_csv(
            pearson.reset_index().rename(columns={"index": "weight"}),
            "iwa_weight_correlations_pearson_matrix.csv",
        ),
        "spearman_matrix": write_csv(
            spearman.reset_index().rename(columns={"index": "weight"}),
            "iwa_weight_correlations_spearman_matrix.csv",
        ),
        "weight_scatter": save_weight_scatter(
            task_summary, OUTPUT_DIR / "iwa_weight_scatter.png"
        ),
        "weight_heatmap": save_weight_correlation_heatmap(
            pearson, OUTPUT_DIR / "iwa_weight_correlation_heatmap.png"
        ),
    }

    counts = {
        "iwa_rows": len(iwa),
        "dwa_rows": len(dwa),
        "dwa_iwas": dwa["IWA ID"].nunique(),
        "task_statement_rows": len(task_statements),
        "task_statement_occupations": task_statements["O*NET-SOC Code"].nunique(),
        "tasks_to_dwas_rows": len(tasks_to_dwas),
        "tasks_to_dwas_occupations": tasks_to_dwas["O*NET-SOC Code"].nunique(),
        "task_detail_rows": len(task_detail),
        "unclassified_task_detail_rows": int(task_detail["task_type"].isna().sum()),
    }

    report_path = write_markdown_report(
        paths,
        counts,
        checks,
        task_summary,
        iwa_occupation_links,
        iwa_counts,
        occupation_counts,
        weight_summary,
        pearson,
    )

    for label, path in paths.items():
        if path.suffix == ".csv":
            print(f"wrote {label}: {len(pd.read_csv(path)):,} rows -> {path}")
        else:
            print(f"wrote {label} -> {path}")
    print(f"wrote report -> {report_path}")


if __name__ == "__main__":
    main()
