"""Cache helpers for ASA GUI pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path


def md5_file(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()
