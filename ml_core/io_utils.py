"""Small filesystem helpers shared across ml_core.

Centralized here so every module writes files the same, safe way: atomic
replace (never leaves a half-written file behind if the process is killed
mid-write, which matters a lot given the platform's hard 3-hour kill) and a
cheap fingerprint used to detect a ValidationPlan being reused against data
it wasn't built for.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(data: bytes, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def atomic_write_text(text: str, path: str | Path, encoding: str = "utf-8") -> None:
    atomic_write_bytes(text.encode(encoding), path)


def atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp_name, index=False, encoding="utf-8")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append one JSON record as a line. Not atomic across concurrent writers
    by design (single-writer append-only log is the expected usage)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def file_fingerprint(path: str | Path) -> str:
    """Cheap fingerprint of a (potentially large) CSV: path + size + a hash of
    first/last 64KB. Good enough to catch "wrong file" / "file changed", not
    meant as a cryptographic content hash of the whole file."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        head = fh.read(65536)
        h.update(head)
        if size > 65536:
            fh.seek(max(0, size - 65536))
            h.update(fh.read(65536))
    return h.hexdigest()[:16]


def config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
