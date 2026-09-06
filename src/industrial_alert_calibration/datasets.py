from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

DatasetPreset = Literal["generic", "metropt", "chiller"]


def load_dataset(path: Path, preset: DatasetPreset) -> pd.DataFrame:
    """Load a supported telemetry file into the pipeline's standard schema."""
    if preset == "generic":
        return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)

    frame = pd.read_csv(path)
    if preset == "metropt":
        if "timestamp" not in frame or "label" not in frame:
            raise ValueError("metropt input must contain timestamp and label columns")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], dayfirst=True, errors="raise", utc=True)
        return frame
    if preset == "chiller":
        if "Date" not in frame or "label" not in frame:
            raise ValueError("chiller input must contain Date and label columns")
        frame = frame.rename(columns={"Date": "timestamp"})
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], format="mixed", dayfirst=True, errors="raise", utc=True
        )
        return frame
    raise ValueError(f"unsupported dataset preset: {preset}")
