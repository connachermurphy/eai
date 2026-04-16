"""Download Anthropic Economic Index release data.

Source: https://huggingface.co/datasets/Anthropic/EconomicIndex
"""

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RELEASE = "release_2026_03_24"
API = f"https://huggingface.co/api/datasets/Anthropic/EconomicIndex/tree/main/{RELEASE}/data"
BASE = f"https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/{RELEASE}/data"

OUT = Path(__file__).resolve().parent.parent / "data" / RELEASE
OUT.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(API) as resp:
    entries = json.load(resp)
names = [Path(e["path"]).name for e in entries if e.get("type") == "file"]

for name in names:
    print(f"downloading {name}...")
    urllib.request.urlretrieve(f"{BASE}/{name}", OUT / name)

(OUT / "metadata.json").write_text(
    json.dumps(
        {
            "release": RELEASE,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "files": names,
        },
        indent=2,
    )
    + "\n"
)
