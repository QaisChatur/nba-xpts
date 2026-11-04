"""Logging, hashing, validation and serialization helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pipeline.config import PROJECT_ROOT


class DataValidationError(ValueError):
    """Raised when an input or intermediate dataset violates its schema or invariants."""


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger("xpts")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return root.getChild(name)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    require(not missing, f"{name} is missing required columns: {missing}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path | str) -> str:
    """Path relative to the project root, for provenance records that must not leak machine paths."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return rel(o)
    if o is pd.NA or o is pd.NaT:
        return None
    raise TypeError(f"not JSON serializable: {type(o)}")


def clean_for_json(obj: Any) -> Any:
    """Recursively replace NaN/inf with None so output is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.floating) and not math.isfinite(float(obj)):
        return None
    try:
        if obj is not None and not isinstance(obj, (str, bytes, dict, list)) and pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_for_json(payload), indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")
    return path


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of JSON-safe dicts."""
    dup = df.columns[df.columns.duplicated()].tolist()
    require(not dup, f"duplicate columns would be silently dropped: {dup}")
    return clean_for_json(df.astype(object).where(df.notna(), None).to_dict(orient="records"))
