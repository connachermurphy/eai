"""Download Eloundou et al. (GPTs are GPTs) occupation-level exposure data.

Source: https://github.com/openai/GPTs-are-GPTs/blob/main/data/occ_level.csv
"""

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/occ_level.csv"
OUT_DIR = Path(__file__).resolve().parent
FILENAME = "occ_level.csv"


def download() -> None:
    """Download occ_level.csv from the GPTs-are-GPTs repo."""
    dest = OUT_DIR / FILENAME
    if dest.exists():
        logger.info("%s already exists, skipping", FILENAME)
        return
    logger.info("Downloading %s...", FILENAME)
    urllib.request.urlretrieve(URL, dest)
    logger.info("Saved to %s", dest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download()
