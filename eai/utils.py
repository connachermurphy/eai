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
    max_keys: int = 20,
    logger: logging.Logger | None = None,
) -> None:
    """Log unmatched keys from a merge.

    Parameters
    ----------
    left_only, right_only : Sets of unmatched keys.
    left_label, right_label : Human-readable names for each side.
    labels : Optional DataFrame mapping keys to descriptive titles.
    key_col : Column in `labels` containing the keys (required if labels is set).
    max_keys : Max unmatched keys to print per side. 0 for unlimited.
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

    def _format_section(unmatched: set[Any], src_label: str, dst_label: str) -> None:
        header = f"In {src_label} but not {dst_label} ({len(unmatched)}):"
        if not unmatched:
            log.info("%s\n  (none)", header)
            return
        sorted_keys = sorted(unmatched)
        show = sorted_keys if max_keys == 0 else sorted_keys[:max_keys]
        lines = [f"  {_format_code(c)}" for c in show]
        if max_keys and len(sorted_keys) > max_keys:
            lines.append(f"  ... and {len(sorted_keys) - max_keys} more")
        log.info("%s\n%s", header, "\n".join(lines))

    _format_section(left_only, left_label, right_label)
    _format_section(right_only, right_label, left_label)
