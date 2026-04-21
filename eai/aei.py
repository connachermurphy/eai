"""Load AEI data across releases and platforms."""

import logging
from pathlib import Path

import pandas as pd

from eai.utils import get_logger

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

RELEASES = {
    "release_2025_09_15": {
        "claude_ai": "aei_cleaned_claude_ai.csv",
        "1p_api": "aei_cleaned_1p_api.csv",
    },
    "release_2026_01_15": {
        "claude_ai": "aei_cleaned_claude_ai.csv",
        "1p_api": "aei_cleaned_1p_api.csv",
    },
    "release_2026_03_24": {
        "claude_ai": "aei_cleaned_claude_ai.csv",
        "1p_api": "aei_cleaned_1p_api.csv",
    },
}

DATA = Path("output")


def load_aei_tasks(
    platforms: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Load cleaned AEI data, stacked across all releases and platforms.

    Returns
    -------
    DataFrame with one row per task-release-platform combination.
    Columns: task, task_key, task_count, {collab}_count, release, platform.
    """
    log = logger or get_logger("aei")
    if platforms is None:
        platforms = ["claude_ai", "1p_api"]

    frames = []
    for release, files in RELEASES.items():
        for platform, filename in files.items():
            if platform not in platforms:
                continue
            path = DATA / release / filename
            if not path.exists():
                log.warning("Missing: %s", path)
                continue
            df = pd.read_csv(path)
            df = df[["task", "task_count"] + [c for c in COUNT_COLS if c in df.columns]]
            df["release"] = release
            df["platform"] = platform
            frames.append(df)
            log.info("Loaded %s / %s: %d tasks", release, platform, len(df))

    if not frames:
        log.error("No AEI data loaded")
        return pd.DataFrame()

    all_data = pd.concat(frames, ignore_index=True)
    all_data["task_key"] = all_data["task"].str.lower().str.strip()

    log.info(
        "AEI stacked: %d rows, %d unique tasks, %d total queries",
        len(all_data),
        all_data["task_key"].nunique(),
        int(all_data["task_count"].sum()),
    )

    return all_data
