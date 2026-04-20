"""Shared helper functions."""

import logging
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name."""
    return logging.getLogger(name)


def merge_with_diagnostics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    how: str = "left",
) -> tuple[pd.DataFrame, set[Any], set[Any]]:
    """Merge two DataFrames and return unmatched keys from each side.

    Returns
    -------
    result : The merged DataFrame.
    left_only : Keys in left but not right.
    right_only : Keys in right but not left.
    """
    result = left.merge(right, on=on, how=how)
    left_only = set(left[on]) - set(right[on])
    right_only = set(right[on]) - set(left[on])
    return result, left_only, right_only


def log_merge_diagnostics(
    left_only: set[Any],
    right_only: set[Any],
    left_label: str = "left",
    right_label: str = "right",
    labels: pd.DataFrame | None = None,
    key_col: str | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Log unmatched keys from a merge.

    Parameters
    ----------
    left_only, right_only : Sets of unmatched keys.
    left_label, right_label : Human-readable names for each side.
    labels : Optional DataFrame mapping keys to descriptive titles.
    key_col : Column in `labels` containing the keys (required if labels is set).
    """
    log = logger or get_logger("merge")

    def _format_code(code: Any) -> str:
        if labels is not None and key_col:
            title_cols = [c for c in labels.columns if "title" in c.lower()]
            if title_cols:
                matches = labels.loc[labels[key_col] == code, title_cols[0]]
                if not matches.empty:
                    return f"{code}  {matches.iloc[0]}"
        return str(code)

    matched = (
        len(left_only | right_only)  # total unique unmatched
        # can't compute matched count without original sets, so just report unmatched
    )

    header = (
        f"In {left_label} but not {right_label} ({len(left_only)}):"
    )
    lines = [f"  {_format_code(c)}" for c in sorted(left_only)]
    log.info("%s\n%s", header, "\n".join(lines) if lines else "  (none)")

    header = (
        f"In {right_label} but not {left_label} ({len(right_only)}):"
    )
    lines = [f"  {_format_code(c)}" for c in sorted(right_only)]
    log.info("%s\n%s", header, "\n".join(lines) if lines else "  (none)")
