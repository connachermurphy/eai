"""Sync committed distribution files from generated outputs."""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

OUTPUT_DIR = Path("output")
DISTRIBUTION_DIR = Path("distribution")

DISTRIBUTION_FILES = (
    "occupations_aei.csv",
    "occupations_eloundou_et_al.csv",
    "occupations_aei_auto_aug_2025_03_27.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy distribution CSVs from output/ into distribution/."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if distribution files are missing or differ from output/.",
    )
    return parser.parse_args()


def distribution_pairs() -> list[tuple[Path, Path]]:
    return [
        (OUTPUT_DIR / filename, DISTRIBUTION_DIR / filename)
        for filename in DISTRIBUTION_FILES
    ]


def check_distribution() -> int:
    errors: list[str] = []
    for source, destination in distribution_pairs():
        if not source.exists():
            errors.append(f"missing source: {source}")
            continue
        if not destination.exists():
            errors.append(f"missing distribution file: {destination}")
            continue
        if not filecmp.cmp(source, destination, shallow=False):
            errors.append(f"stale distribution file: {destination}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("distribution files are up to date")
    return 0


def sync_distribution() -> int:
    DISTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)
    missing_sources = [
        source for source, _ in distribution_pairs() if not source.exists()
    ]
    if missing_sources:
        for source in missing_sources:
            print(f"missing source: {source}", file=sys.stderr)
        return 1

    for source, destination in distribution_pairs():
        shutil.copy2(source, destination)
        print(f"copied {source} -> {destination}")
    return 0


def main() -> int:
    args = parse_args()
    if args.check:
        return check_distribution()
    return sync_distribution()


if __name__ == "__main__":
    raise SystemExit(main())
