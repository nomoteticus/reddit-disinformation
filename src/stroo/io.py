"""Reading and writing the pipeline's CSV files the same way on every platform.

Reddit text can contain carriage returns, so rows are always terminated by a
single line feed, on Windows too.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, lineterminator="\n", **kwargs)


def write_csv(table: pd.DataFrame, path: Path, **kwargs) -> None:
    table.to_csv(path, index=False, lineterminator="\n", **kwargs)
