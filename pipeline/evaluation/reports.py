"""Markdown rendering helpers for reports."""

from __future__ import annotations

import math

import pandas as pd


def fmt(v, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "–"
        return f"{v:.{digits}f}"
    return str(v)


def md_table(df: pd.DataFrame, digits: int = 4, index: bool = False) -> str:
    d = df.reset_index() if index else df
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v, digits).replace("|", "/") for v in row) + " |")
    return "\n".join(lines)
