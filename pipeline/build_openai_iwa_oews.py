# ruff: noqa: E501
"""Merge OpenAI IWA message shares with O*NET IWA links and OEWS employment."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from eai.codebook import update_codebook
from eai.utils import get_logger

log = get_logger(__name__)

DEFAULT_OEWS_YEAR = 2024

OPENAI_INPUT_DIR = Path("input") / "oai_260616" / "data-download-csv"
IWA_MAPPING_DIR = Path("output") / "onet" / "iwa_mapping"
OUTPUT_DIR = Path("output") / "openai_iwa_oews"
DATA_DIR = Path("output")

IWA_DETAIL_PATH = IWA_MAPPING_DIR / "iwa_to_onet_soc_via_tasks_detail.csv"

OPENAI_IWA_FILES = [
    {
        "file_name": "usa_share_of_messages_by_onet_iwa_month.csv",
        "openai_measure": "us_all_messages_iwa_share",
        "formal_definition": "P(iwa_cleaned | month)",
        "denominator": "U.S. consumer ChatGPT messages",
    },
    {
        "file_name": "usa_share_of_work_related_messages_by_onet_iwa_month.csv",
        "openai_measure": "us_work_related_messages_iwa_share",
        "formal_definition": "P(iwa_cleaned | month, work_related)",
        "denominator": "U.S. work-related consumer ChatGPT messages",
    },
]

MEAN_SUMMARY_MEASURE_COLUMNS = {
    "us_all_messages_iwa_share": "mean_us_all_messages_iwa_share",
    "us_work_related_messages_iwa_share": "mean_us_work_related_messages_iwa_share",
}

LINK_COUNT_COLUMNS = [
    "link_task_count",
    "link_task_dwa_link_count",
    "link_dwa_count",
    "link_core_task_count",
    "link_supplemental_task_count",
    "link_unclassified_task_count",
    "link_classified_task_count",
]

PANEL_COLUMNS = [
    "openai_measure",
    "month",
    "iwa_id",
    "iwa_title",
    "soc_2018",
    "title_2018",
    "group_id",
    "openai_iwa_share_of_messages",
    "employment_weight_within_iwa",
    "soc_2018_apportioned_share_of_messages",
    "apportionment_method",
    "oews_tot_emp_imputed",
    "oews_tot_emp_adjusted",
    "oews_emp_was_imputed",
    "oews_a_mean",
    "oews_a_median",
    "oews_broad_match",
    "oews_soc_2018_broad",
    "soc_2018_count_for_iwa",
    "iwa_count_for_soc_2018",
    "onet_soc_count",
    "onet_soc_codes",
    "onet_soc_titles",
    *LINK_COUNT_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Merge OpenAI Signals O*NET IWA shares with O*NET IWA-SOC links "
            "and OEWS employment, using employment apportionment only."
        )
    )
    parser.add_argument(
        "--oews-year",
        type=int,
        default=DEFAULT_OEWS_YEAR,
        help=f"OEWS year to load from output/oews (default: {DEFAULT_OEWS_YEAR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for generated outputs (default: {OUTPUT_DIR}).",
    )
    return parser.parse_args()


def join_unique(values: pd.Series) -> str:
    """Join non-empty unique values in sorted order."""
    items = sorted({str(value) for value in values.dropna() if str(value)})
    return "; ".join(items)


def parse_periods(values: pd.Series) -> pd.Series:
    """Parse O*NET month/year strings."""
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


def fmt(value: object, digits: int = 4) -> str:
    """Format values for the Markdown report."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(
    df: pd.DataFrame, columns: list[str], limit: int | None = None
) -> str:
    """Render a small DataFrame as a Markdown table."""
    table = df[columns].copy()
    if limit is not None:
        table = table.head(limit)
    headers = [column.replace("_", " ") for column in columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def load_soc_2018() -> pd.DataFrame:
    """Load the existing SOC 2018 universe used by the repo."""
    soc_2018 = pd.read_csv(DATA_DIR / "onet" / "soc_2018_to_group.csv")
    broad_codes = soc_2018[soc_2018["soc_2018"].str[-1] == "0"]
    if not broad_codes.empty:
        raise ValueError(
            "SOC 2018 universe should contain detailed codes only, but found "
            f"{len(broad_codes)} broad codes."
        )
    return soc_2018


def build_oews_lookup(oews_year: int, soc_2018: pd.DataFrame) -> pd.DataFrame:
    """Build the OEWS SOC 2018 lookup using the repo's exact plus broad matching rule."""
    oews_path = DATA_DIR / "oews" / f"national_M{oews_year}_dl.csv"
    if not oews_path.exists():
        raise FileNotFoundError(f"Missing OEWS file: {oews_path}")

    oews_raw = pd.read_csv(oews_path)
    oews = oews_raw[oews_raw["o_group"] == "detailed"][
        ["occ_code", "occ_title", "tot_emp", "a_mean", "a_median"]
    ].copy()
    oews = oews.rename(columns={"occ_code": "soc_2018"})
    for column in ["tot_emp", "a_mean", "a_median"]:
        oews[column] = pd.to_numeric(oews[column], errors="coerce")

    universe_codes = set(soc_2018["soc_2018"])
    oews_codes = set(oews["soc_2018"])

    exact = oews[oews["soc_2018"].isin(universe_codes)].copy()
    exact["broad_match"] = False
    exact["soc_2018_broad"] = pd.NA
    exact["tot_emp_adjusted"] = exact["tot_emp"]

    unmatched_codes = universe_codes - oews_codes
    unmatched_oews = oews[oews["soc_2018"].isin(oews_codes - universe_codes)]

    broad_map = pd.DataFrame({"soc_2018": sorted(unmatched_codes)})
    broad_map["soc_2018_broad"] = broad_map["soc_2018"].str[:6] + "0"
    broad_matched = broad_map.merge(
        unmatched_oews.rename(columns={"soc_2018": "soc_2018_broad"}),
        on="soc_2018_broad",
        how="left",
    )
    broad_matched = broad_matched[broad_matched["tot_emp"].notna()].copy()
    n_per_broad = broad_matched.groupby("soc_2018_broad")["soc_2018"].transform(
        "nunique"
    )
    broad_matched["broad_match"] = True
    broad_matched["tot_emp_adjusted"] = broad_matched["tot_emp"] / n_per_broad

    lookup_cols = [
        "soc_2018",
        "occ_title",
        "tot_emp",
        "a_mean",
        "a_median",
        "broad_match",
        "soc_2018_broad",
        "tot_emp_adjusted",
    ]
    lookup = pd.concat(
        [exact[lookup_cols], broad_matched[lookup_cols]], ignore_index=True
    )
    lookup = lookup.merge(
        soc_2018[["soc_2018", "title_2018", "group_id"]],
        on="soc_2018",
        how="left",
        validate="1:1",
    )
    lookup = lookup.rename(
        columns={
            "occ_title": "oews_occ_title",
            "tot_emp": "oews_tot_emp",
            "a_mean": "oews_a_mean",
            "a_median": "oews_a_median",
            "broad_match": "oews_broad_match",
            "soc_2018_broad": "oews_soc_2018_broad",
            "tot_emp_adjusted": "oews_tot_emp_adjusted",
        }
    )
    log.info(
        "OEWS %d lookup: %d exact + %d broad = %d SOC 2018 rows",
        oews_year,
        len(exact),
        len(broad_matched),
        len(lookup),
    )
    return lookup


def build_iwa_soc_links(soc_2018: pd.DataFrame) -> pd.DataFrame:
    """Collapse O*NET-SOC IWA links to SOC 2018 for OEWS apportionment."""
    detail = pd.read_csv(IWA_DETAIL_PATH)
    detail["soc_2018"] = detail["onet_soc_code"].str[:7]

    missing_soc = sorted(set(detail["soc_2018"]) - set(soc_2018["soc_2018"]))
    if missing_soc:
        raise ValueError(
            "O*NET 30.2 link SOC prefixes should align to SOC 2018, but missing: "
            f"{missing_soc[:20]}"
        )

    group_keys = ["iwa_id", "iwa_title", "soc_2018"]
    summary = (
        detail.groupby(group_keys, dropna=False)
        .agg(
            onet_soc_count=("onet_soc_code", "nunique"),
            onet_soc_codes=("onet_soc_code", join_unique),
            onet_soc_titles=("occupation_title", join_unique),
            link_task_count=("task_id", "nunique"),
            link_task_dwa_link_count=("task_id", "size"),
            link_dwa_count=("dwa_id", "nunique"),
            first_mapping_update=("date", min_period),
            latest_mapping_update=("date", max_period),
            domain_sources=("domain_source", join_unique),
        )
        .reset_index()
    )

    task_counts = (
        detail.drop_duplicates(group_keys + ["task_type_clean", "task_id"])
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
            "Core": "link_core_task_count",
            "Supplemental": "link_supplemental_task_count",
            "Unclassified": "link_unclassified_task_count",
        }
    )
    task_counts["link_classified_task_count"] = (
        task_counts["link_core_task_count"]
        + task_counts["link_supplemental_task_count"]
    )

    links = summary.merge(task_counts, on=group_keys, how="left", validate="1:1")
    links = links.merge(
        soc_2018[["soc_2018", "title_2018", "group_id"]],
        on="soc_2018",
        how="left",
        validate="m:1",
    )
    links["soc_2018_count_for_iwa"] = links.groupby("iwa_id")["soc_2018"].transform(
        "nunique"
    )
    links["iwa_count_for_soc_2018"] = links.groupby("soc_2018")["iwa_id"].transform(
        "nunique"
    )

    ordered = [
        "iwa_id",
        "iwa_title",
        "soc_2018",
        "title_2018",
        "group_id",
        "soc_2018_count_for_iwa",
        "iwa_count_for_soc_2018",
        "onet_soc_count",
        "onet_soc_codes",
        "onet_soc_titles",
        *LINK_COUNT_COLUMNS,
        "first_mapping_update",
        "latest_mapping_update",
        "domain_sources",
    ]
    links = links[ordered].sort_values(["iwa_id", "soc_2018"]).reset_index(drop=True)
    log.info(
        "IWA-SOC 2018 links: %d rows, %d IWAs, %d SOC 2018 occupations",
        len(links),
        links["iwa_id"].nunique(),
        links["soc_2018"].nunique(),
    )
    return links


def add_employment_weights(
    links: pd.DataFrame, oews_lookup: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Merge OEWS and add the only active apportionment weight: employment."""
    oews_cols = [
        "soc_2018",
        "oews_occ_title",
        "oews_tot_emp",
        "oews_a_mean",
        "oews_a_median",
        "oews_broad_match",
        "oews_soc_2018_broad",
        "oews_tot_emp_adjusted",
    ]
    linked = links.merge(oews_lookup[oews_cols], on="soc_2018", how="left")
    linked["oews_broad_match"] = linked["oews_broad_match"].fillna(False)
    linked["oews_emp_was_imputed"] = linked["oews_tot_emp_adjusted"].isna()

    linked_soc_employment = linked[
        ["soc_2018", "oews_tot_emp_adjusted"]
    ].drop_duplicates()
    median_emp = linked_soc_employment.loc[
        linked_soc_employment["oews_tot_emp_adjusted"].notna(),
        "oews_tot_emp_adjusted",
    ].median()
    linked["oews_tot_emp_imputed"] = linked["oews_tot_emp_adjusted"].fillna(median_emp)

    linked["employment_total_for_iwa"] = linked.groupby("iwa_id")[
        "oews_tot_emp_imputed"
    ].transform("sum")
    linked["employment_weight_within_iwa"] = np.where(
        linked["employment_total_for_iwa"] > 0,
        linked["oews_tot_emp_imputed"] / linked["employment_total_for_iwa"],
        np.nan,
    )
    weight_sums = linked.groupby("iwa_id")["employment_weight_within_iwa"].sum(
        min_count=1
    )
    bad_weights = weight_sums[~np.isclose(weight_sums, 1.0, atol=1e-9)]
    if not bad_weights.empty:
        raise ValueError(
            "Employment weights do not sum to 1 for IWAs: "
            f"{bad_weights.index.tolist()[:10]}"
        )

    log.info(
        "Employment coverage: %d/%d IWA-SOC links have OEWS data; imputed unique-SOC median %.0f for %d links",
        linked["oews_tot_emp_adjusted"].notna().sum(),
        len(linked),
        median_emp,
        linked["oews_emp_was_imputed"].sum(),
    )
    return linked, float(median_emp)


def build_soc_2018_employment_universe(
    soc_2018: pd.DataFrame,
    oews_lookup: pd.DataFrame,
    median_emp: float,
) -> pd.DataFrame:
    """Build one row per SOC 2018 occupation with OEWS fields for summary outputs."""
    oews_cols = [
        "soc_2018",
        "oews_occ_title",
        "oews_tot_emp",
        "oews_a_mean",
        "oews_a_median",
        "oews_broad_match",
        "oews_soc_2018_broad",
        "oews_tot_emp_adjusted",
    ]
    universe = soc_2018[["soc_2018", "title_2018", "group_id"]].merge(
        oews_lookup[oews_cols],
        on="soc_2018",
        how="left",
        validate="1:1",
    )
    universe["oews_broad_match"] = universe["oews_broad_match"].fillna(False)
    universe["oews_emp_was_imputed"] = universe["oews_tot_emp_adjusted"].isna()
    universe["oews_tot_emp_imputed"] = universe["oews_tot_emp_adjusted"].fillna(
        median_emp
    )
    return universe[
        [
            "soc_2018",
            "title_2018",
            "group_id",
            "oews_tot_emp_imputed",
            "oews_tot_emp_adjusted",
            "oews_a_mean",
            "oews_a_median",
            "oews_broad_match",
            "oews_soc_2018_broad",
            "oews_emp_was_imputed",
        ]
    ]


def load_openai_iwa_files() -> pd.DataFrame:
    """Load and stack the OpenAI Signals IWA month files."""
    rows = []
    for spec in OPENAI_IWA_FILES:
        path = OPENAI_INPUT_DIR / spec["file_name"]
        if not path.exists():
            raise FileNotFoundError(f"Missing OpenAI Signals file: {path}")
        df = pd.read_csv(path)
        df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m-%d")
        df = df.rename(
            columns={
                "iwa_cleaned": "iwa_id",
                "share_of_messages": "openai_iwa_share_of_messages",
            }
        )
        df["openai_measure"] = spec["openai_measure"]
        df["openai_source_file"] = spec["file_name"]
        df["formal_definition"] = spec["formal_definition"]
        df["denominator"] = spec["denominator"]
        rows.append(df)
    openai = pd.concat(rows, ignore_index=True)
    openai = openai[
        [
            "openai_measure",
            "month",
            "iwa_id",
            "openai_iwa_share_of_messages",
            "formal_definition",
            "denominator",
            "openai_source_file",
        ]
    ]
    log.info(
        "OpenAI IWA data: %d rows, %d measures, %d months, %d IWA labels",
        len(openai),
        openai["openai_measure"].nunique(),
        openai["month"].nunique(),
        openai["iwa_id"].nunique(),
    )
    return openai


def build_iwa_month(
    openai: pd.DataFrame, linked: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add IWA-level mapping and employment coverage to OpenAI IWA rows."""
    iwa_lookup = (
        linked.groupby(["iwa_id", "iwa_title"], as_index=False)
        .agg(
            soc_2018_count_for_iwa=("soc_2018", "nunique"),
            onet_soc_count_for_iwa=("onet_soc_count", "sum"),
            oews_employment_total_for_iwa=("oews_tot_emp_imputed", "sum"),
            oews_missing_soc_2018_count=("oews_emp_was_imputed", "sum"),
            link_task_count=("link_task_count", "sum"),
            link_task_dwa_link_count=("link_task_dwa_link_count", "sum"),
            link_dwa_count=("link_dwa_count", "sum"),
        )
        .reset_index(drop=True)
    )
    iwa_month = openai.merge(iwa_lookup, on="iwa_id", how="left")
    iwa_month["is_mapped_to_onet_iwa"] = iwa_month["iwa_title"].notna()
    unmatched = iwa_month[~iwa_month["is_mapped_to_onet_iwa"]].copy()
    non_other_unmatched = unmatched[unmatched["iwa_id"] != "Other IWA"]
    if not non_other_unmatched.empty:
        raise ValueError(
            "OpenAI IWA labels are missing from the O*NET mapping: "
            f"{sorted(non_other_unmatched['iwa_id'].unique())[:20]}"
        )
    return iwa_month, unmatched


def build_apportioned_panel(
    openai: pd.DataFrame,
    linked: pd.DataFrame,
    soc_2018_employment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apportion OpenAI IWA shares across linked SOC 2018 occupations."""
    mapped_openai = openai[openai["iwa_id"].isin(set(linked["iwa_id"]))].copy()
    panel = mapped_openai.merge(linked, on="iwa_id", how="left", validate="m:m")
    panel["soc_2018_apportioned_share_of_messages"] = (
        panel["openai_iwa_share_of_messages"] * panel["employment_weight_within_iwa"]
    )
    panel["apportionment_method"] = "employment"

    weight_check = (
        panel.groupby(["openai_measure", "month", "iwa_id"], as_index=False)
        .agg(
            openai_iwa_share_of_messages=("openai_iwa_share_of_messages", "first"),
            employment_weight_sum=("employment_weight_within_iwa", "sum"),
            apportioned_share_sum=(
                "soc_2018_apportioned_share_of_messages",
                "sum",
            ),
        )
        .assign(
            allocation_residual=lambda df: (
                df["apportioned_share_sum"] - df["openai_iwa_share_of_messages"]
            )
        )
    )
    bad_iwa = weight_check[
        (~np.isclose(weight_check["employment_weight_sum"], 1.0, atol=1e-9))
        | (~np.isclose(weight_check["allocation_residual"], 0.0, atol=1e-12))
    ]
    if not bad_iwa.empty:
        raise ValueError(
            "IWA allocation failed validation for rows: "
            f"{bad_iwa.head().to_dict(orient='records')}"
        )

    observed_summary = panel.groupby(
        ["openai_measure", "month", "soc_2018"], as_index=False
    ).agg(
        soc_2018_apportioned_share_of_messages=(
            "soc_2018_apportioned_share_of_messages",
            "sum",
        ),
        iwa_count_contributing=("iwa_id", "nunique"),
    )
    measure_months = openai[["openai_measure", "month"]].drop_duplicates()
    summary = measure_months.merge(soc_2018_employment, how="cross").merge(
        observed_summary,
        on=["openai_measure", "month", "soc_2018"],
        how="left",
        validate="1:1",
    )
    summary["soc_2018_apportioned_share_of_messages"] = summary[
        "soc_2018_apportioned_share_of_messages"
    ].fillna(0)
    summary["iwa_count_contributing"] = (
        summary["iwa_count_contributing"].fillna(0).astype(int)
    )
    summary = summary.sort_values(["openai_measure", "month", "soc_2018"]).reset_index(
        drop=True
    )
    summary = summary[
        [
            "openai_measure",
            "month",
            "soc_2018",
            "title_2018",
            "group_id",
            "soc_2018_apportioned_share_of_messages",
            "iwa_count_contributing",
            "oews_tot_emp_imputed",
            "oews_tot_emp_adjusted",
            "oews_emp_was_imputed",
            "oews_a_mean",
            "oews_a_median",
            "oews_broad_match",
            "oews_soc_2018_broad",
        ]
    ]

    return panel[PANEL_COLUMNS], summary, weight_check


def build_month_checks(
    openai: pd.DataFrame, panel: pd.DataFrame, unmatched: pd.DataFrame
) -> pd.DataFrame:
    """Compare source IWA shares with apportioned occupation shares by month."""
    source = (
        openai.groupby(["openai_measure", "month"], as_index=False)[
            "openai_iwa_share_of_messages"
        ]
        .sum()
        .rename(columns={"openai_iwa_share_of_messages": "source_iwa_share_sum"})
    )
    unmatched_sum = (
        unmatched.groupby(["openai_measure", "month"], as_index=False)[
            "openai_iwa_share_of_messages"
        ]
        .sum()
        .rename(columns={"openai_iwa_share_of_messages": "unmatched_iwa_share_sum"})
    )
    apportioned = (
        panel.groupby(["openai_measure", "month"], as_index=False)[
            "soc_2018_apportioned_share_of_messages"
        ]
        .sum()
        .rename(
            columns={
                "soc_2018_apportioned_share_of_messages": "apportioned_soc_2018_share_sum"
            }
        )
    )
    checks = source.merge(unmatched_sum, on=["openai_measure", "month"], how="left")
    checks = checks.merge(apportioned, on=["openai_measure", "month"], how="left")
    checks["unmatched_iwa_share_sum"] = checks["unmatched_iwa_share_sum"].fillna(0)
    checks["mapped_iwa_share_sum"] = (
        checks["source_iwa_share_sum"] - checks["unmatched_iwa_share_sum"]
    )
    checks["allocation_residual"] = (
        checks["apportioned_soc_2018_share_sum"] - checks["mapped_iwa_share_sum"]
    )
    bad_months = checks[~np.isclose(checks["allocation_residual"], 0.0, atol=1e-12)]
    if not bad_months.empty:
        raise ValueError(
            "Month allocation checks failed: "
            f"{bad_months.head().to_dict(orient='records')}"
        )
    return checks[
        [
            "openai_measure",
            "month",
            "source_iwa_share_sum",
            "mapped_iwa_share_sum",
            "unmatched_iwa_share_sum",
            "apportioned_soc_2018_share_sum",
            "allocation_residual",
        ]
    ]


def build_mean_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Average monthly SOC 2018 shares and return one row per occupation."""
    usage_long = summary.groupby(["soc_2018", "openai_measure"], as_index=False).agg(
        mean_share=(
            "soc_2018_apportioned_share_of_messages",
            "mean",
        ),
        n_months=("month", "nunique"),
        first_month=("month", "min"),
        last_month=("month", "max"),
    )

    expected_months = summary["month"].nunique()
    incomplete = usage_long[usage_long["n_months"] != expected_months]
    if not incomplete.empty:
        raise ValueError(
            "Mean summary expected balanced monthly coverage; incomplete rows: "
            f"{incomplete.head().to_dict(orient='records')}"
        )

    usage_wide = usage_long.pivot(
        index="soc_2018",
        columns="openai_measure",
        values="mean_share",
    ).reset_index()
    usage_wide = usage_wide.rename(columns=MEAN_SUMMARY_MEASURE_COLUMNS)
    expected_usage_cols = list(MEAN_SUMMARY_MEASURE_COLUMNS.values())
    missing_cols = [col for col in expected_usage_cols if col not in usage_wide.columns]
    if missing_cols:
        raise ValueError(
            f"Mean summary is missing expected measure columns: {missing_cols}"
        )

    metadata = (
        summary.groupby("soc_2018", as_index=False)
        .agg(
            title_2018=("title_2018", "first"),
            group_id=("group_id", "first"),
            oews_tot_emp_imputed=("oews_tot_emp_imputed", "first"),
            oews_tot_emp_adjusted=("oews_tot_emp_adjusted", "first"),
            oews_emp_was_imputed=("oews_emp_was_imputed", "first"),
            oews_a_mean=("oews_a_mean", "first"),
            oews_a_median=("oews_a_median", "first"),
            oews_broad_match=("oews_broad_match", "first"),
            oews_soc_2018_broad=("oews_soc_2018_broad", "first"),
        )
        .reset_index(drop=True)
    )
    mean_summary = metadata.merge(usage_wide, on="soc_2018", how="left", validate="1:1")
    return mean_summary[
        [
            "soc_2018",
            "title_2018",
            "group_id",
            *expected_usage_cols,
            "oews_tot_emp_imputed",
            "oews_tot_emp_adjusted",
            "oews_emp_was_imputed",
            "oews_a_mean",
            "oews_a_median",
            "oews_broad_match",
            "oews_soc_2018_broad",
        ]
    ].sort_values("soc_2018")


def build_apportionment_sensitivity(
    panel: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare employment apportionment with diagnostic non-output alternatives."""
    baseline = summary[
        [
            "openai_measure",
            "month",
            "soc_2018",
            "soc_2018_apportioned_share_of_messages",
        ]
    ].copy()
    full_grid = baseline[["openai_measure", "month", "soc_2018"]].drop_duplicates()
    baseline_mean = baseline.groupby(
        ["openai_measure", "soc_2018"], as_index=False
    ).agg(
        employment_mean_share=(
            "soc_2018_apportioned_share_of_messages",
            "mean",
        )
    )

    diagnostics = [
        {
            "diagnostic_apportionment": "equal",
            "weight_basis": "Equal weight across linked SOC 2018 occupations",
            "count_column": None,
        },
        {
            "diagnostic_apportionment": "link_task_count",
            "weight_basis": "Distinct O*NET task count on each IWA-SOC link",
            "count_column": "link_task_count",
        },
        {
            "diagnostic_apportionment": "link_task_dwa_link_count",
            "weight_basis": "O*NET task-DWA edge count on each IWA-SOC link",
            "count_column": "link_task_dwa_link_count",
        },
        {
            "diagnostic_apportionment": "link_dwa_count",
            "weight_basis": "Distinct DWA count on each IWA-SOC link",
            "count_column": "link_dwa_count",
        },
    ]

    rows = []
    group_cols = ["openai_measure", "month", "iwa_id"]
    for diagnostic in diagnostics:
        working = panel[
            [
                "openai_measure",
                "month",
                "iwa_id",
                "soc_2018",
                "openai_iwa_share_of_messages",
                *LINK_COUNT_COLUMNS,
            ]
        ].copy()
        n_soc = working.groupby(group_cols)["soc_2018"].transform("nunique")
        equal_weight = np.where(n_soc > 0, 1.0 / n_soc, np.nan)
        count_column = diagnostic["count_column"]
        if count_column is None:
            working["diagnostic_weight"] = equal_weight
        else:
            denominator = working.groupby(group_cols)[count_column].transform("sum")
            working["diagnostic_weight"] = np.where(
                denominator > 0,
                working[count_column] / denominator,
                equal_weight,
            )

        working["diagnostic_share"] = (
            working["openai_iwa_share_of_messages"] * working["diagnostic_weight"]
        )
        alt_month = working.groupby(
            ["openai_measure", "month", "soc_2018"], as_index=False
        ).agg(diagnostic_share=("diagnostic_share", "sum"))
        alt_month = full_grid.merge(
            alt_month,
            on=["openai_measure", "month", "soc_2018"],
            how="left",
            validate="1:1",
        )
        alt_month["diagnostic_share"] = alt_month["diagnostic_share"].fillna(0)
        alt_mean = alt_month.groupby(
            ["openai_measure", "soc_2018"], as_index=False
        ).agg(diagnostic_mean_share=("diagnostic_share", "mean"))
        comparison = baseline_mean.merge(
            alt_mean,
            on=["openai_measure", "soc_2018"],
            how="left",
            validate="1:1",
        )
        comparison["abs_difference"] = (
            comparison["employment_mean_share"] - comparison["diagnostic_mean_share"]
        ).abs()
        for measure, sub in comparison.groupby("openai_measure"):
            rows.append(
                {
                    "openai_measure": measure,
                    "diagnostic_apportionment": diagnostic["diagnostic_apportionment"],
                    "weight_basis": diagnostic["weight_basis"],
                    "n_occupations": len(sub),
                    "pearson_vs_employment": sub["employment_mean_share"].corr(
                        sub["diagnostic_mean_share"],
                        method="pearson",
                    ),
                    "spearman_vs_employment": sub["employment_mean_share"].corr(
                        sub["diagnostic_mean_share"],
                        method="spearman",
                    ),
                    "mean_abs_difference": sub["abs_difference"].mean(),
                    "max_abs_difference": sub["abs_difference"].max(),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["openai_measure", "diagnostic_apportionment"]
    )


def write_report(
    output_dir: Path,
    paths: dict[str, Path],
    oews_year: int,
    median_emp: float,
    openai: pd.DataFrame,
    iwa_month: pd.DataFrame,
    unmatched: pd.DataFrame,
    linked: pd.DataFrame,
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    month_checks: pd.DataFrame,
    apportionment_sensitivity: pd.DataFrame,
) -> Path:
    """Write a concise Markdown report."""
    coverage_rows = []
    for measure, sub in iwa_month.groupby("openai_measure"):
        mapped_iwa_labels = sub.loc[sub["is_mapped_to_onet_iwa"], "iwa_id"].nunique()
        iwa_labels = sub["iwa_id"].nunique()
        coverage_rows.append(
            {
                "openai_measure": measure,
                "months": sub["month"].nunique(),
                "iwa_labels": iwa_labels,
                "mapped_iwa_labels": mapped_iwa_labels,
                "unmatched_iwa_labels": iwa_labels - mapped_iwa_labels,
                "rows": len(sub),
            }
        )
    openai_coverage = pd.DataFrame(coverage_rows)

    employment_coverage = pd.DataFrame(
        [
            {
                "metric": "IWA-SOC links",
                "value": len(linked),
            },
            {
                "metric": "IWAs in O*NET 30.2 mapping",
                "value": linked["iwa_id"].nunique(),
            },
            {
                "metric": "SOC 2018 occupations in links",
                "value": linked["soc_2018"].nunique(),
            },
            {
                "metric": "Links with direct/broad OEWS employment",
                "value": int(linked["oews_tot_emp_adjusted"].notna().sum()),
            },
            {
                "metric": "Links imputed to median employment",
                "value": int(linked["oews_emp_was_imputed"].sum()),
            },
            {
                "metric": "Median employment used for imputation",
                "value": median_emp,
            },
            {
                "metric": "OpenAI rows",
                "value": len(openai),
            },
            {
                "metric": "Apportioned link-month rows",
                "value": len(panel),
            },
            {
                "metric": "SOC 2018 month summary rows",
                "value": len(summary),
            },
            {
                "metric": "SOC 2018 mean summary rows",
                "value": len(mean_summary),
            },
        ]
    )

    top_unmatched = (
        unmatched.groupby(["openai_measure", "iwa_id"], as_index=False)
        .agg(
            months=("month", "nunique"),
            mean_share=("openai_iwa_share_of_messages", "mean"),
            max_share=("openai_iwa_share_of_messages", "max"),
        )
        .sort_values(["openai_measure", "max_share"], ascending=[True, False])
    )
    month_check_summary = (
        month_checks.groupby("openai_measure", as_index=False)
        .agg(
            months=("month", "nunique"),
            source_share_min=("source_iwa_share_sum", "min"),
            source_share_max=("source_iwa_share_sum", "max"),
            unmatched_share_mean=("unmatched_iwa_share_sum", "mean"),
            max_abs_allocation_residual=(
                "allocation_residual",
                lambda x: x.abs().max(),
            ),
        )
        .sort_values("openai_measure")
    )
    zero_usage_summary = pd.DataFrame(
        [
            {
                "usage_column": column,
                "zero_occupation_count": int((mean_summary[column] == 0).sum()),
                "nonzero_occupation_count": int((mean_summary[column] != 0).sum()),
            }
            for column in MEAN_SUMMARY_MEASURE_COLUMNS.values()
        ]
    )
    mean_summary_window = pd.DataFrame(
        [
            {
                "months": summary["month"].nunique(),
                "first_month": summary["month"].min(),
                "last_month": summary["month"].max(),
            }
        ]
    )

    report = f"""# OpenAI IWA to OEWS Employment Apportionment

Generated {date.today().isoformat()}.

## Sources

- OpenAI Signals CSVs in `{
        OPENAI_INPUT_DIR
    }`. The bundled README defines `share_of_messages` as a differentially private weighted share in `[0, 1]`; the two O*NET IWA files are monthly U.S. IWA shares, one among all U.S. messages and one among work-related U.S. messages.
- O*NET 30.2 files in `input/db_30_2_excel`. The O*NET 30.2 dictionary documents `IWA Reference`, `DWA Reference`, and `Tasks to DWAs`; it also states that the current O*NET-SOC 2019 taxonomy is based on the transition to the 2018 SOC.
- OEWS May {oews_year} national data in `output/oews/national_M{oews_year}_dl.csv`.

## Method

1. Collapse the existing O*NET 30.2 task-DWA-IWA detail mapping to one row per `iwa_id` x six-digit `soc_2018`. This avoids duplicating OEWS employment across multiple O*NET-SOC suboccupations under the same SOC code.
2. Merge OEWS to SOC 2018 using exact SOC matches plus the repo's simple trailing-zero broad-code fallback. Exact detailed SOC matches are used first. When OEWS only reports a broader detailed-family code ending in `0`, employment is divided equally across matching SOC 2018 detailed occupations in the repo's SOC universe. Other OEWS aggregate reporting codes are not allocated through the SOC 2010 to SOC 2018 crosswalk or `group_id`; for example, `25-2052` is not split across `25-2055` and `25-2056` in this pipeline.
3. Impute missing linked SOC 2018 employment to the median employment among unique linked SOC 2018 occupations with OEWS employment. This avoids letting occupations with more IWA links receive extra weight in the imputation statistic.
4. Within each IWA, compute `employment_weight_within_iwa = oews_tot_emp_imputed / sum(oews_tot_emp_imputed)`.
5. For each OpenAI measure/month/IWA, allocate only by employment: `soc_2018_apportioned_share_of_messages = openai_iwa_share_of_messages * employment_weight_within_iwa`.
6. Average the monthly SOC 2018 apportioned shares across all available months to produce the mean summary.

No alternate exposure outputs are implemented. The link count columns remain in the static link and panel outputs as diagnostics for later methodological changes.

## Outputs

{
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "artifact": paths["linked"].name,
                        "rows": len(linked),
                        "description": "Static IWA x SOC 2018 link table with OEWS employment and employment weights.",
                    },
                    {
                        "artifact": paths["iwa_month"].name,
                        "rows": len(iwa_month),
                        "description": "Stacked OpenAI IWA month data with mapping and employment coverage fields.",
                    },
                    {
                        "artifact": paths["panel"].name,
                        "rows": len(panel),
                        "description": "IWA x SOC 2018 x month panel with employment-apportioned OpenAI shares.",
                    },
                    {
                        "artifact": paths["summary"].name,
                        "rows": len(summary),
                        "description": "SOC 2018 x month summary, summing apportioned shares across IWAs.",
                    },
                    {
                        "artifact": paths["mean_summary"].name,
                        "rows": len(mean_summary),
                        "description": "SOC 2018 summary with separate mean usage columns for each OpenAI IWA measure.",
                    },
                    {
                        "artifact": paths["unmatched"].name,
                        "rows": len(unmatched),
                        "description": "OpenAI IWA rows that cannot be mapped to O*NET 30.2, currently the privacy bucket `Other IWA`.",
                    },
                    {
                        "artifact": paths["checks"].name,
                        "rows": len(month_checks),
                        "description": "Month-level allocation checks.",
                    },
                    {
                        "artifact": paths["weight_check"].name,
                        "rows": panel[["openai_measure", "month", "iwa_id"]]
                        .drop_duplicates()
                        .shape[0],
                        "description": "IWA-level checks that employment weights sum to one and allocated shares recover source IWA shares.",
                    },
                ]
            ),
            ["artifact", "rows", "description"],
        )
    }

## Coverage

{markdown_table(employment_coverage, ["metric", "value"])}

### OpenAI Measures

{
        markdown_table(
            openai_coverage,
            [
                "openai_measure",
                "months",
                "iwa_labels",
                "mapped_iwa_labels",
                "unmatched_iwa_labels",
                "rows",
            ],
        )
    }

### Month Checks

{
        markdown_table(
            month_check_summary,
            [
                "openai_measure",
                "months",
                "source_share_min",
                "source_share_max",
                "unmatched_share_mean",
                "max_abs_allocation_residual",
            ],
        )
    }

### Apportionment Sensitivity Diagnostic

The exposure outputs use employment apportionment only. As a diagnostic, the table
below compares the resulting occupation-level mean shares with equal allocation
and three link-count-based ways of distributing IWA shares across linked
occupations. These diagnostic allocations are not saved as exposure outputs. The
Spearman correlations suggest that rank order is fairly stable, while the lower
Pearson correlations and maximum absolute differences show that the level of the
occupation exposure measure is meaningfully sensitive to the apportionment
assumption.

{
        markdown_table(
            apportionment_sensitivity,
            [
                "openai_measure",
                "diagnostic_apportionment",
                "n_occupations",
                "pearson_vs_employment",
                "spearman_vs_employment",
                "mean_abs_difference",
                "max_abs_difference",
            ],
        )
    }

### Zero Usage In Mean Summary

{
        markdown_table(
            zero_usage_summary,
            ["usage_column", "zero_occupation_count", "nonzero_occupation_count"],
        )
    }

### Mean Summary Window

{
        markdown_table(
            mean_summary_window,
            ["months", "first_month", "last_month"],
        )
    }

### Unmatched OpenAI Labels

{
        markdown_table(
            top_unmatched,
            ["openai_measure", "iwa_id", "months", "mean_share", "max_share"],
        )
    }

## Outstanding Questions

- The OpenAI release groups rare IWA labels under `Other IWA`; this bucket is intentionally left unallocated because it does not correspond to a specific O*NET IWA.
- OEWS is SOC 2018 while O*NET 30.2 is O*NET-SOC 2019. This script allocates at six-digit SOC 2018 and keeps the contributing O*NET-SOC suboccupation codes as diagnostics.
- The occupation-level exposure measure is sensitive to how IWA-level OpenAI shares are apportioned across occupations linked to the same IWA. Employment apportionment is the active choice here, but sensitivity to this choice should be kept in mind when interpreting levels or wage correlations.
- Median employment imputation now uses unique linked SOC 2018 occupations rather than link-expanded IWA-SOC rows. The imputed SOCs should still be reviewed if these outputs become inputs to a production apportionment.
- The OEWS merge intentionally does not use the SOC 2010 to SOC 2018 crosswalk or `group_id` to allocate non-trailing-zero aggregate reporting codes. Known cases such as `25-2052` are left unmatched to SOC 2018 children and handled by the median-imputation rule above.
- The OpenAI shares are differentially private rounded shares, so monthly source totals can be close to, but not exactly, 1.
"""
    path = output_dir / "iwa_openai_oews_report.md"
    path.write_text(report, encoding="utf-8")
    return path


GROUP_ID_DEF = (
    "Connected-component ID of the bipartite SOC 2010<->2018 crosswalk graph; "
    "non-singleton groups mark one-to-many or many-to-many crosswalk "
    "relationships."
)
SOC_2018_ID_COLS = [
    ("soc_2018", "Six-digit SOC 2018 occupation code."),
    ("title_2018", "SOC 2018 occupation title."),
    ("group_id", GROUP_ID_DEF),
]
IWA_ID_COLS = [
    ("iwa_id", "O*NET 30.2 Intermediate Work Activity (IWA) identifier."),
    ("iwa_title", "IWA title."),
]
OEWS_SOC_2018_COLS = [
    ("oews_occ_title", "OEWS national occupation title for the matched code."),
    (
        "oews_tot_emp",
        "Raw OEWS national total employment for the exact SOC 2018 code (NA "
        "when missing or suppressed).",
    ),
    ("oews_a_mean", "OEWS annual mean wage."),
    ("oews_a_median", "OEWS annual median wage."),
    (
        "oews_broad_match",
        "True when employment came from the trailing-zero broad-code fallback "
        "rather than an exact SOC 2018 match.",
    ),
    (
        "oews_soc_2018_broad",
        "SOC 2018 broad code (fifth digit zeroed) used for the employment "
        "fallback; NA for exact matches.",
    ),
    (
        "oews_tot_emp_adjusted",
        "Employment after the broad-code fallback: the exact-match value, or "
        "the broad code's employment split equally across its detailed child "
        "codes without their own OEWS row.",
    ),
    (
        "oews_emp_was_imputed",
        "True when oews_tot_emp_adjusted is missing, so oews_tot_emp_imputed "
        "holds the median-employment fill value.",
    ),
    (
        "oews_tot_emp_imputed",
        "oews_tot_emp_adjusted with missing employment filled with the median "
        "across unique linked SOC 2018 occupations. Used for the employment "
        "weights.",
    ),
]
IWA_LINK_COLS = [
    (
        "soc_2018_count_for_iwa",
        "Number of distinct SOC 2018 codes linked to the IWA.",
    ),
    (
        "iwa_count_for_soc_2018",
        "Number of distinct IWAs linked to the SOC 2018 code.",
    ),
    (
        "onet_soc_count",
        "Number of O*NET-SOC 2019 occupation codes collapsed into this "
        "six-digit SOC 2018 link.",
    ),
    ("onet_soc_codes", "Semicolon-joined O*NET-SOC 2019 codes for the link."),
    ("onet_soc_titles", "Semicolon-joined O*NET-SOC 2019 titles for the link."),
    (
        "link_task_count",
        "Distinct O*NET task IDs connecting the IWA to the occupation(s) in the link.",
    ),
    (
        "link_task_dwa_link_count",
        "Task-to-DWA link rows behind the link (a task can map to several "
        "DWAs under one IWA).",
    ),
    ("link_dwa_count", "Distinct DWA IDs under the IWA for the link."),
    ("link_core_task_count", "Distinct linked tasks typed Core in O*NET."),
    (
        "link_supplemental_task_count",
        "Distinct linked tasks typed Supplemental in O*NET.",
    ),
    (
        "link_unclassified_task_count",
        "Distinct linked tasks with no O*NET task type.",
    ),
    (
        "link_classified_task_count",
        "Core plus supplemental linked task count.",
    ),
    (
        "first_mapping_update",
        "Earliest O*NET Tasks-to-DWAs mapping date contributing to the link (YYYY-MM).",
    ),
    (
        "latest_mapping_update",
        "Latest O*NET Tasks-to-DWAs mapping date contributing to the link (YYYY-MM).",
    ),
    (
        "domain_sources",
        "Joined unique O*NET domain sources contributing to the link.",
    ),
]
OPENAI_MEASURE_COLS = [
    (
        "openai_measure",
        "Which OpenAI Signals series the row comes from: "
        "us_all_messages_iwa_share (all U.S. consumer ChatGPT messages) or "
        "us_work_related_messages_iwa_share (work-related U.S. messages).",
    ),
    ("month", "Calendar month of the OpenAI observation (YYYY-MM-DD)."),
    (
        "openai_iwa_share_of_messages",
        "OpenAI-reported share of messages classified to the IWA in the "
        "month, as a proportion in [0, 1]. Shares are differentially private "
        "rounded values, so monthly totals are close to but not exactly 1.",
    ),
    (
        "formal_definition",
        "Formal definition of the measure from the OpenAI release metadata "
        "(e.g., P(iwa | month)).",
    ),
    ("denominator", "Message population the share is taken over."),
    ("openai_source_file", "OpenAI Signals source CSV file name."),
]


def _oews_cols(*names: str) -> list[tuple[str, str]]:
    """Pick OEWS column definitions by name, in the given order."""
    defs = dict(OEWS_SOC_2018_COLS)
    return [(name, defs[name]) for name in names]


PANEL_OEWS_COLS = _oews_cols(
    "oews_tot_emp_imputed",
    "oews_tot_emp_adjusted",
    "oews_emp_was_imputed",
    "oews_a_mean",
    "oews_a_median",
    "oews_broad_match",
    "oews_soc_2018_broad",
)

IWA_MONTH_COLS = (
    OPENAI_MEASURE_COLS[:2]
    + [IWA_ID_COLS[0]]
    + OPENAI_MEASURE_COLS[2:]
    + [IWA_ID_COLS[1]]
    + [
        (
            "soc_2018_count_for_iwa",
            "Number of distinct SOC 2018 codes linked to the IWA.",
        ),
        (
            "onet_soc_count_for_iwa",
            "Number of O*NET-SOC 2019 codes linked to the IWA.",
        ),
        (
            "oews_employment_total_for_iwa",
            "Sum of oews_tot_emp_imputed across SOC 2018 codes linked to the IWA.",
        ),
        (
            "oews_missing_soc_2018_count",
            "Number of linked SOC 2018 codes whose employment was imputed "
            "(no exact or broad OEWS match).",
        ),
        ("link_task_count", "Distinct O*NET task IDs across the IWA's links."),
        (
            "link_task_dwa_link_count",
            "Task-to-DWA link rows across the IWA's links.",
        ),
        ("link_dwa_count", "Distinct DWA IDs across the IWA's links."),
        (
            "is_mapped_to_onet_iwa",
            "True when the OpenAI IWA label matches an O*NET 30.2 IWA; False "
            "for the `Other IWA` privacy bucket, which is left unallocated.",
        ),
    ]
)


def write_output_codebook(output_dir: Path) -> None:
    """Write the codebook for all outputs in this directory."""
    update_codebook(
        output_dir / "codebook.md",
        section="openai_iwa_oews",
        title="OpenAI IWA/OEWS occupation measures",
        source="pipeline/build_openai_iwa_oews.py",
        intro=(
            "OpenAI Signals IWA-level message shares allocated to SOC 2018 "
            "occupations by employment within each IWA. All shares are "
            "proportions in [0, 1]."
        ),
        files=[
            {
                "name": "iwa_soc2018_employment_links.csv",
                "description": (
                    "Static IWA by SOC 2018 link table with OEWS employment "
                    "weights. O*NET-SOC 2019 codes are collapsed to six-digit "
                    "SOC 2018."
                ),
                "columns": IWA_ID_COLS
                + SOC_2018_ID_COLS
                + IWA_LINK_COLS
                + OEWS_SOC_2018_COLS
                + [
                    (
                        "employment_total_for_iwa",
                        "Sum of oews_tot_emp_imputed across SOC 2018 codes "
                        "linked to the IWA.",
                    ),
                    (
                        "employment_weight_within_iwa",
                        "oews_tot_emp_imputed / employment_total_for_iwa; "
                        "sums to 1 across the IWA's occupations.",
                    ),
                ],
            },
            {
                "name": "openai_iwa_month.csv",
                "description": (
                    "Stacked OpenAI IWA-month rows with mapping and "
                    "employment coverage fields."
                ),
                "columns": IWA_MONTH_COLS,
            },
            {
                "name": "openai_iwa_unmatched.csv",
                "description": (
                    "OpenAI IWA-month rows not allocated to occupations "
                    "(currently only the `Other IWA` privacy bucket). Same "
                    "columns as openai_iwa_month.csv."
                ),
                "columns": IWA_MONTH_COLS,
            },
            {
                "name": "openai_iwa_soc2018_month_panel.csv",
                "description": (
                    "IWA by SOC 2018 by month link panel with apportioned shares."
                ),
                "columns": OPENAI_MEASURE_COLS[:2]
                + IWA_ID_COLS
                + SOC_2018_ID_COLS
                + [OPENAI_MEASURE_COLS[2]]
                + [
                    (
                        "employment_weight_within_iwa",
                        "oews_tot_emp_imputed / total imputed employment "
                        "across SOC 2018 codes linked to the IWA; sums to 1 "
                        "within IWA.",
                    ),
                    (
                        "soc_2018_apportioned_share_of_messages",
                        "openai_iwa_share_of_messages * "
                        "employment_weight_within_iwa: the occupation's "
                        "employment-apportioned slice of the IWA share.",
                    ),
                    (
                        "apportionment_method",
                        "Constant 'employment'; the only allocation rule produced.",
                    ),
                ]
                + PANEL_OEWS_COLS
                + IWA_LINK_COLS,
            },
            {
                "name": "openai_soc2018_month_summary.csv",
                "description": (
                    "SOC 2018 by month panel: apportioned OpenAI shares "
                    "summed across IWAs, on the full SOC 2018 universe "
                    "(zero-filled when no IWA contributes)."
                ),
                "columns": OPENAI_MEASURE_COLS[:2]
                + SOC_2018_ID_COLS
                + [
                    (
                        "soc_2018_apportioned_share_of_messages",
                        "Sum of the occupation's apportioned IWA shares for "
                        "the measure-month; 0 when no linked IWA has usage.",
                    ),
                    (
                        "iwa_count_contributing",
                        "Number of distinct IWAs contributing to the "
                        "occupation-month cell.",
                    ),
                ]
                + PANEL_OEWS_COLS,
            },
            {
                "name": "openai_soc2018_mean_summary.csv",
                "description": (
                    "One row per SOC 2018 occupation with mean apportioned "
                    "OpenAI shares across all available months."
                ),
                "columns": SOC_2018_ID_COLS
                + [
                    (
                        "mean_us_all_messages_iwa_share",
                        "Mean of soc_2018_apportioned_share_of_messages "
                        "across months for the all-U.S.-messages measure.",
                    ),
                    (
                        "mean_us_work_related_messages_iwa_share",
                        "Mean of soc_2018_apportioned_share_of_messages "
                        "across months for the work-related-U.S.-messages "
                        "measure.",
                    ),
                ]
                + PANEL_OEWS_COLS,
            },
            {
                "name": "openai_iwa_oews_month_checks.csv",
                "description": (
                    "Allocation validation per measure-month; the build fails "
                    "if any residual is nonzero."
                ),
                "columns": OPENAI_MEASURE_COLS[:2]
                + [
                    (
                        "source_iwa_share_sum",
                        "Sum of OpenAI IWA shares in the source data.",
                    ),
                    (
                        "mapped_iwa_share_sum",
                        "Source sum minus the unmatched (Other IWA) share.",
                    ),
                    (
                        "unmatched_iwa_share_sum",
                        "Share mass in unallocated IWA labels.",
                    ),
                    (
                        "apportioned_soc_2018_share_sum",
                        "Sum of apportioned occupation shares.",
                    ),
                    (
                        "allocation_residual",
                        "apportioned_soc_2018_share_sum - "
                        "mapped_iwa_share_sum; should be ~0.",
                    ),
                ],
            },
            {
                "name": "openai_iwa_oews_weight_checks.csv",
                "description": (
                    "Allocation validation per IWA-measure-month; the build "
                    "fails if weights do not sum to 1."
                ),
                "columns": OPENAI_MEASURE_COLS[:2]
                + [IWA_ID_COLS[0]]
                + [OPENAI_MEASURE_COLS[2]]
                + [
                    (
                        "employment_weight_sum",
                        "Sum of employment_weight_within_iwa across the IWA's "
                        "occupations; should be 1.",
                    ),
                    (
                        "apportioned_share_sum",
                        "Sum of the IWA's apportioned occupation shares.",
                    ),
                    (
                        "allocation_residual",
                        "apportioned_share_sum - "
                        "openai_iwa_share_of_messages; should be ~0.",
                    ),
                ],
            },
        ],
    )
    log.info("Updated codebook: %s", output_dir / "codebook.md")


def main() -> None:
    """Build all outputs."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    soc_2018 = load_soc_2018()
    oews_lookup = build_oews_lookup(args.oews_year, soc_2018)
    links = build_iwa_soc_links(soc_2018)
    linked, median_emp = add_employment_weights(links, oews_lookup)
    soc_2018_employment = build_soc_2018_employment_universe(
        soc_2018,
        oews_lookup,
        median_emp,
    )
    openai = load_openai_iwa_files()
    iwa_month, unmatched = build_iwa_month(openai, linked)
    panel, summary, weight_check = build_apportioned_panel(
        openai,
        linked,
        soc_2018_employment,
    )
    mean_summary = build_mean_summary(summary)
    month_checks = build_month_checks(openai, panel, unmatched)
    apportionment_sensitivity = build_apportionment_sensitivity(panel, summary)

    paths = {
        "linked": args.output_dir / "iwa_soc2018_employment_links.csv",
        "iwa_month": args.output_dir / "openai_iwa_month.csv",
        "unmatched": args.output_dir / "openai_iwa_unmatched.csv",
        "panel": args.output_dir / "openai_iwa_soc2018_month_panel.csv",
        "summary": args.output_dir / "openai_soc2018_month_summary.csv",
        "mean_summary": args.output_dir / "openai_soc2018_mean_summary.csv",
        "checks": args.output_dir / "openai_iwa_oews_month_checks.csv",
        "weight_check": args.output_dir / "openai_iwa_oews_weight_checks.csv",
    }
    linked.to_csv(paths["linked"], index=False)
    iwa_month.to_csv(paths["iwa_month"], index=False)
    unmatched.to_csv(paths["unmatched"], index=False)
    panel.to_csv(paths["panel"], index=False)
    summary.to_csv(paths["summary"], index=False)
    mean_summary.to_csv(paths["mean_summary"], index=False)
    month_checks.to_csv(paths["checks"], index=False)
    weight_check.to_csv(paths["weight_check"], index=False)
    write_output_codebook(args.output_dir)
    report_path = write_report(
        args.output_dir,
        paths,
        args.oews_year,
        median_emp,
        openai,
        iwa_month,
        unmatched,
        linked,
        panel,
        summary,
        mean_summary,
        month_checks,
        apportionment_sensitivity,
    )

    log.info("Saved report: %s", report_path)
    for label, path in paths.items():
        log.info("Saved %s: %s", label, path)


if __name__ == "__main__":
    main()
