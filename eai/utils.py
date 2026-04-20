"""Shared helper functions."""

import pandas as pd


def merge_with_diagnostics(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    how: str = "left",
    left_label: str = "left",
    right_label: str = "right",
) -> pd.DataFrame:
    """Merge two DataFrames and print diagnostics about unmatched keys."""
    result = left.merge(right, on=on, how=how)

    left_keys = set(left[on])
    right_keys = set(right[on])
    in_left_not_right = sorted(left_keys - right_keys)
    in_right_not_left = sorted(right_keys - left_keys)

    print(f"\nMerge diagnostics ({left_label} x {right_label} on '{on}'):")
    print(f"  {left_label}: {len(left_keys)} keys")
    print(f"  {right_label}: {len(right_keys)} keys")
    print(f"  Matched: {len(left_keys & right_keys)}")

    if in_left_not_right:
        print(
            f"\n  In {left_label} but not {right_label}"
            f" ({len(in_left_not_right)}):"
        )
        for code in in_left_not_right:
            print(f"    {code}")

    if in_right_not_left:
        print(
            f"\n  In {right_label} but not {left_label}"
            f" ({len(in_right_not_left)}):"
        )
        for code in in_right_not_left:
            print(f"    {code}")

    return result
