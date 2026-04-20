"""Download Anthropic Economic Index release data from Hugging Face.

Source: https://huggingface.co/datasets/Anthropic/EconomicIndex
"""

import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main"
API_URL = "https://huggingface.co/api/datasets/Anthropic/EconomicIndex/tree/main"

# Registry of releases and their data paths relative to the release root.
# Older releases store data in data/intermediate/, newer ones in data/.
RELEASES = {
    "release_2025_09_15": "data/intermediate",
    "release_2026_01_15": "data/intermediate",
    "release_2026_03_24": "data",
}

OUT_BASE = Path(__file__).resolve().parent.parent / "data"


def download_release(release: str, data_subpath: str) -> None:
    """Download all files for a single release."""
    api = f"{API_URL}/{release}/{data_subpath}"
    out_dir = OUT_BASE / release
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching file list for %s", release)
    with urllib.request.urlopen(api) as resp:
        entries = json.load(resp)
    names = [Path(e["path"]).name for e in entries if e.get("type") == "file"]

    base = f"{BASE_URL}/{release}/{data_subpath}"
    for name in names:
        dest = out_dir / name
        if dest.exists():
            logger.info("  %s (already exists, skipping)", name)
            continue
        logger.info("  downloading %s...", name)
        urllib.request.urlretrieve(f"{base}/{name}", dest)

    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "release": release,
                "data_subpath": data_subpath,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "files": names,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("Done: %s (%d files)", release, len(names))


def download_all() -> None:
    """Download all registered releases."""
    for release, subpath in RELEASES.items():
        download_release(release, subpath)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_all()
