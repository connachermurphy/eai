"""Clean AEI raw data: filter to GLOBAL, pivot collaboration counts wide.

Processes all releases × platforms (Claude AI and 1P API), outputting one
cleaned CSV per release-platform combination.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

COLLAB_TYPES = [
    "directive",
    "feedback_loop",
    "learning",
    "none",
    "not_classified",
    "task_iteration",
    "validation",
]
COUNT_COLS = [f"{t}_count" for t in COLLAB_TYPES]
PCT_COLS = [f"{t}_pct" for t in COLLAB_TYPES]

# Registry: release -> {platform_key: raw_filename}
RELEASES = {
    "release_2025_09_15": {
        "claude_ai": "aei_raw_claude_ai_2025-08-04_to_2025-08-11.csv",
        "1p_api": "aei_raw_1p_api_2025-08-04_to_2025-08-11.csv",
    },
    "release_2026_01_15": {
        "claude_ai": "aei_raw_claude_ai_2025-11-13_to_2025-11-20.csv",
        "1p_api": "aei_raw_1p_api_2025-11-13_to_2025-11-20.csv",
    },
    "release_2026_03_24": {
        "claude_ai": "aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv",
        "1p_api": "aei_raw_1p_api_2026-02-05_to_2026-02-12.csv",
    },
}


def clean_aei(raw_path: Path, release: str, platform: str) -> pd.DataFrame:
    """Clean a single AEI raw file: filter to GLOBAL, pivot collaboration wide.

    Parameters
    ----------
    raw_path : Path to the raw CSV file
    release : Release identifier (for logging)
    platform : Platform key ("claude_ai" or "1p_api")

    Returns
    -------
    DataFrame with columns: task, task_count, task_pct, {collab_type}_{count,pct}
    """
    logger.info("[%s / %s] Loading %s", release, platform, raw_path.name)
    raw = pd.read_csv(raw_path)
    logger.info("  Raw rows: %d", len(raw))

    # Filter to GLOBAL
    raw = raw[raw["geo_id"] == "GLOBAL"]
    logger.info("  After GLOBAL filter: %d rows", len(raw))

    # --- Collaboration data ---
    df = raw[raw["facet"] == "onet_task::collaboration"].copy()
    if df.empty:
        logger.warning("  No onet_task::collaboration rows — skipping")
        return pd.DataFrame()

    # --- Task usage totals (from onet_task facet) ---
    usage = raw[raw["facet"] == "onet_task"].copy()
    usage["var_short"] = usage["variable"].str.replace("onet_task_", "")
    usage_dupes = usage.duplicated(subset=["cluster_name", "var_short"], keep=False)
    if usage_dupes.any():
        logger.error(
            "  Duplicate (cluster_name, var_short) pairs found — aborting this file"
        )
        return pd.DataFrame()

    usage_wide = (
        usage.pivot_table(
            index="cluster_name",
            columns="var_short",
            values="value",
            aggfunc="first",
        )
        .rename(columns={"count": "task_count", "pct": "task_pct"})
        .reset_index()
        .rename(columns={"cluster_name": "task"})
    )

    # --- Parse cluster_name ---
    df["task"] = df["cluster_name"].str.rsplit("::", n=1).str[0]
    df["collaboration_type"] = df["cluster_name"].str.rsplit("::", n=1).str[1]
    df["collaboration_type"] = df["collaboration_type"].str.replace(" ", "_")

    # Shorten variable names
    df["var_short"] = df["variable"].str.replace("onet_task_collaboration_", "")
    df["col_name"] = df["collaboration_type"] + "_" + df["var_short"]

    # --- Pivot Wide ---
    dupes = df.duplicated(subset=["task", "col_name"], keep=False)
    if dupes.any():
        logger.error("  Duplicate (task, col_name) pairs — aborting this file")
        return pd.DataFrame()

    wide = (
        df.pivot_table(index="task", columns="col_name", values="value", aggfunc="first")
        .fillna(0)
        .reset_index()
    )

    # --- Merge task usage totals ---
    usage_wide["task_merge_key"] = usage_wide["task"].str.lower().str.strip()
    wide["task_merge_key"] = wide["task"].str.lower().str.strip()
    wide_keys = set(wide["task_merge_key"])
    usage_keys = set(usage_wide["task_merge_key"])

    usage_unmatched = usage_keys - wide_keys
    if usage_unmatched:
        logger.info(
            "  Usage tasks not in collaboration (dropped): %d", len(usage_unmatched)
        )

    aei_unmatched = wide_keys - usage_keys
    if aei_unmatched:
        logger.info(
            "  Collaboration tasks not in usage (null task_count): %d",
            len(aei_unmatched),
        )

    wide = wide.merge(
        usage_wide[["task_merge_key", "task_count", "task_pct"]],
        on="task_merge_key",
        how="left",
    )
    wide = wide.drop(columns=["task_merge_key"])

    # --- Order columns ---
    # Handle case where not all collab types are present
    available_count = [c for c in COUNT_COLS if c in wide.columns]
    available_pct = [c for c in PCT_COLS if c in wide.columns]
    wide = wide[["task", "task_count", "task_pct"] + available_count + available_pct]

    # --- Validation ---
    total_count = int(wide["task_count"].sum())
    logger.info("  Total conversations (after facet filter): %s", f"{total_count:,}")
    top = wide.nlargest(3, "task_count")[["task", "task_count"]]
    for _, row in top.iterrows():
        logger.info(
            "    max: %s (%s)",
            f"{int(row['task_count']):,}",
            row["task"][:80],
        )

    logger.info("  Output: %d tasks", len(wide))
    return wide


def clean_all() -> None:
    """Process all releases and platforms."""
    for release, platforms in RELEASES.items():
        for platform, filename in platforms.items():
            raw_path = DATA / release / filename
            if not raw_path.exists():
                logger.warning(
                    "[%s / %s] Raw file not found: %s — skipping",
                    release,
                    platform,
                    raw_path,
                )
                continue

            wide = clean_aei(raw_path, release, platform)
            if wide.empty:
                continue

            out_path = DATA / release / f"aei_cleaned_{platform}.csv"
            wide.to_csv(out_path, index=False)
            logger.info("[%s / %s] Wrote %s", release, platform, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    clean_all()
